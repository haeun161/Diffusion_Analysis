"""
sd3_hook.py — verify / causal_path 양쪽이 공유하는 hook 로직
==============================================================

블록 내부 흐름:

  [block 입력] hidden_states  (batch, 4096, 1536), float16
       │
       ├── ① L_minus_1 캡처         ← block 시작 시점
       │
       ▼
  norm1 + attention (image ↔ text joint attention)
       │
       ▼
  hidden_states = hidden_states + gate * attn_output
  (= L_minus_1 + delta_attn  =  L_after_attn)
       │
       ├── ② L_after_attn 캡처      ← norm2.register_forward_pre_hook
       │       norm2 실행 직전 → args_in[0] = norm2 입력 = L_after_attn
       │
       ▼
  hidden_states = hidden_states + gate * ff(norm2(hidden_states))
  (= L_after_attn + delta_mlp  =  L_output)
       │
       └── ③ L_output 캡처          ← block_output[1] (image stream)

  반환값 구조:
    block  0~22 (context_pre_only=False): tuple(text[333], image[4096]) → [1]
    block 23    (context_pre_only=True):  tensor(image[4096])           → 자체

  batch 구성:
    CFG(Classifier-Free Guidance)로 batch=2 로 들어옴
    [0] = unconditional  (negative prompt)
    [1] = conditional    (실제 prompt)
    → 둘 다 저장 (uncond/cond 각각 분석 가능)

  분해식:
    delta_attn = L_after_attn - L_minus_1
    delta_mlp  = L_output     - L_after_attn
    항등식:  L_minus_1 + delta_attn + delta_mlp == L_output

  f32 변환 규칙:
    ★ 반드시 .float() 먼저, 뺄셈은 나중에
    f16끼리 뺄셈 후 f32 변환하면 rounding 오차가 delta에 고정됨
    f32 변환 후 뺄셈하면 항등식이 machine epsilon(~1e-7) 수준으로 성립
"""

from functools import wraps


def make_block_hook(original_forward, block_idx, blk, on_capture):
    """
    JointTransformerBlock.forward를 패치하는 함수.

    Parameters
    ----------
    original_forward : callable
        패치 전 원본 block.forward
    block_idx : int
        블록 번호 (0~23)
    blk : nn.Module
        블록 인스턴스 (norm2 접근용)
    on_capture : callable(block_idx, L_minus_1, L_after_attn, L_output, batch_size)
        캡처된 텐서 3개를 전달받는 콜백.
        텐서 shape: (batch, 4096, 1536), dtype: float16, device: CPU
        batch_size: int (보통 2, CFG)
        콜백 안에서 .float() 변환 및 저장 로직을 구현.

    Returns
    -------
    patched_forward : callable
        block.forward에 할당할 패치된 함수.
        _is_patched = True 속성이 붙어있음 (중복 패치 방지용).
    """

    @wraps(original_forward)
    def patched_forward(hidden_states, encoder_hidden_states, temb, *args, **kwargs):

        # ──────────────────────────────────────────────────────
        # ① L_minus_1: block에 들어온 image token stream
        #   shape : (batch, 4096, 1536)
        #   dtype : float16  (모델 연산 dtype 그대로)
        #   detach: 그래디언트 추적 끊기
        #   clone : 이후 in-place 연산에 의해 값이 바뀌지 않도록 복사
        # ──────────────────────────────────────────────────────
        L_minus_1 = hidden_states.detach().clone()  # (batch, 4096, 1536), float16

        # ──────────────────────────────────────────────────────
        # ② L_after_attn: norm2 입력 = attention residual 적용 후, MLP 이전
        #
        #   왜 norm2 pre_hook 인가?
        #   블록 내부 순서:
        #     hidden_states += gate * attn_output    ← attention residual 적용
        #     hidden_states += gate * ff(norm2(...)) ← MLP
        #   norm2의 입력 = "attention은 끝, MLP는 아직" = L_after_attn
        #
        #   register_forward_pre_hook:
        #     norm2.forward() 실행 직전 콜백 호출
        #     args_in[0] = norm2에 전달되는 hidden_states = L_after_attn
        # ──────────────────────────────────────────────────────
        norm2_input_buffer = {}

        def _capture_norm2_input(module, args_in):
            # 한 번만 캡처
            if "L_after_attn" not in norm2_input_buffer and args_in:
                norm2_input_buffer["L_after_attn"] = args_in[0].detach().clone()

        norm2_hook_handle = blk.norm2.register_forward_pre_hook(_capture_norm2_input)

        try:
            block_output = original_forward(
                hidden_states, encoder_hidden_states, temb, *args, **kwargs
            )
        finally:
            # 에러 여부 무관하게 반드시 제거
            # (제거 안 하면 이후 모든 forward에서 계속 호출됨)
            norm2_hook_handle.remove()

        # ──────────────────────────────────────────────────────
        # ③ L_output: block 최종 출력의 image token stream
        #
        #   block 0~22 (context_pre_only=False):
        #     tuple → (encoder_hidden_states[333], hidden_states[4096])
        #     image stream = block_output[1]
        #
        #   block 23 (context_pre_only=True):
        #     tensor → hidden_states[4096] 만 반환
        #     image stream = block_output 자체
        # ──────────────────────────────────────────────────────
        if isinstance(block_output, tuple):
            L_output = block_output[1].detach()   # image stream
        else:
            L_output = block_output.detach()      # context_pre_only 블록

        # ──────────────────────────────────────────────────────
        # 콜백 호출: 캡처된 텐서를 caller에게 전달
        # batch 전체(uncond + cond) 그대로 전달
        # → 분리/저장은 콜백 안에서 결정
        # ──────────────────────────────────────────────────────
        if "L_after_attn" in norm2_input_buffer:
            on_capture(
                block_idx    = block_idx,
                L_minus_1    = L_minus_1.cpu(),                              # (batch, 4096, 1536), f16
                L_after_attn = norm2_input_buffer["L_after_attn"].cpu(),     # (batch, 4096, 1536), f16
                L_output     = L_output.cpu(),                               # (batch, 4096, 1536), f16
                batch_size   = L_minus_1.shape[0],
            )

        return block_output

    patched_forward._is_patched = True   # 중복 패치 방지용 마킹
    return patched_forward


def patch_transformer(transformer, on_capture):
    """
    모든 JointTransformerBlock을 패치.

    Parameters
    ----------
    transformer : SD3Transformer2DModel
    on_capture  : callable  (make_block_hook의 on_capture와 동일 시그니처)

    Returns
    -------
    already_patched_count : int
    """
    already_patched = 0
    for block_idx, block in enumerate(transformer.transformer_blocks):
        if getattr(block.forward, "_is_patched", False):
            already_patched += 1
            continue
        block.forward = make_block_hook(block.forward, block_idx, block, on_capture)
    return already_patched


def attach_step_tracker(transformer, on_step):
    """
    transformer.forward를 감싸서 denoising step 번호를 추적.

    Parameters
    ----------
    on_step : callable(step_idx: int) -> bool
        step_idx를 받아 "이 step을 캡처할지" bool 반환.
        True면 해당 step에서 on_capture 콜백이 활성화됨.

    Returns
    -------
    step_counter : list[int]
        [0] 인덱스로 현재 step 번호에 접근/리셋 가능.
    """
    if getattr(transformer.forward, "_is_step_tracked", False):
        return None   # 이미 부착됨

    original_forward = transformer.forward
    step_counter = [0]
    # hook_active를 외부에서 제어하기 위해 리스트로 감쌈
    hook_active = [False]

    @wraps(original_forward)
    def step_tracked_forward(*args, **kwargs):
        hook_active[0] = on_step(step_counter[0])
        result = original_forward(*args, **kwargs)
        step_counter[0] += 1
        return result

    step_tracked_forward._is_step_tracked = True
    step_tracked_forward._hook_active = hook_active   # 외부 접근용
    transformer.forward = step_tracked_forward
    return step_counter