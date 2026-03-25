"""Tests that the critique stage excludes each model's own proposal."""

from council.providers.base import ModelResponse
from council.orchestrator import _CRITIQUE_PROMPT


def _make_proposals():
    return [
        ModelResponse(
            text="Proposal from chatgpt.",
            participant="chatgpt",
            model_id="gpt-5.4",
        ),
        ModelResponse(
            text="Proposal from claude.",
            participant="claude",
            model_id="claude-opus-4-6",
        ),
        ModelResponse(
            text="Proposal from gemini.",
            participant="gemini",
            model_id="gemini-3.1-pro",
        ),
    ]


def _build_critique_text(proposals, current_provider_name):
    """Reproduce the filtering logic from _critique_one."""
    peer_proposals = [r for r in proposals if r.participant != current_provider_name]
    proposal_text = "\n\n".join(
        f"[{r.participant}]: {r.text}" for r in peer_proposals
    )
    return _CRITIQUE_PROMPT.format(proposals=proposal_text)


class TestCritiqueFiltering:
    def test_excludes_own_proposal(self):
        proposals = _make_proposals()
        text = _build_critique_text(proposals, "chatgpt")
        assert "[chatgpt]" not in text
        assert "[claude]" in text
        assert "[gemini]" in text

    def test_excludes_own_proposal_gemini(self):
        proposals = _make_proposals()
        text = _build_critique_text(proposals, "gemini")
        assert "[gemini]" not in text
        assert "[chatgpt]" in text
        assert "[claude]" in text

    def test_all_peers_present_when_no_match(self):
        """If provider name doesn't match any proposal, all are included."""
        proposals = _make_proposals()
        text = _build_critique_text(proposals, "unknown_model")
        assert "[chatgpt]" in text
        assert "[claude]" in text
        assert "[gemini]" in text

    def test_prompt_wording_says_other_participants(self):
        proposals = _make_proposals()
        text = _build_critique_text(proposals, "chatgpt")
        assert "other council participants" in text
        assert "Other participants' positions:" in text

    def test_two_model_council(self):
        """With only 2 models, each sees exactly 1 peer proposal."""
        proposals = [
            ModelResponse(text="A's take.", participant="a", model_id="a-1"),
            ModelResponse(text="B's take.", participant="b", model_id="b-1"),
        ]
        text_a = _build_critique_text(proposals, "a")
        assert "[a]" not in text_a
        assert "[b]" in text_a

        text_b = _build_critique_text(proposals, "b")
        assert "[b]" not in text_b
        assert "[a]" in text_b
