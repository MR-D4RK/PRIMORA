"""M3 foundation: domain-independent pronoun/recall/episode safety tests."""
import pytest

from PRIMORA.prime.context import ContextManager
from PRIMORA.prime.context.discourse import resolve_preference_reference
from PRIMORA.prime.context.episode_retrieval import parse_episode_question
from PRIMORA.prime.context.state import extract_candidates
from PRIMORA.prime.context.mind import Mind
from PRIMORA.aurora.loop import CognitiveLoop
from PRIMORA.aurora.brain import BrainResult


@pytest.mark.parametrize('previous,subject', [
    ('im currently listening to a song which is big room', 'big room'),
    ('I am reading a novel named Seventh Horizon', 'Seventh Horizon'),
    ('We are testing a device called Quanta Reader', 'Quanta Reader'),
    ('I am using software titled Indigo Glass', 'Indigo Glass'),
    ('I am working on an unfamiliar gadget called Zero Bloom', 'Zero Bloom'),
])
def test_unseen_subjects_use_identical_grammar(tmp_path, previous, subject):
    m = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    m.reconcile([{'role': 'user', 'content': previous, 'timestamp': 1720000000},
                 {'role': 'assistant', 'content': 'Tell me more.', 'timestamp': 1720000001},
                 {'role': 'user', 'content': 'I love it', 'timestamp': 1720000002}],
                'talk')
    next_session = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    states = [s for s in next_session._get_store().fetch_states(scope='global', status='current') if s.concept.startswith('likes for ')]
    assert len(states) == 1
    assert states[0].concept == 'likes for ' + subject.casefold()
    assert states[0].value.casefold() == 'loves ' + subject.casefold()
    assert len(states[0].source_ids) == 2
    answer = next_session.verified_likes_history('what have i told you i like?')
    assert subject.casefold() in answer.casefold()
    assert previous in answer and 'I love it' in answer
    assert 'I love it' in answer
    assert 'source' not in answer or 'message' in answer


def test_no_unsupported_pronoun_or_cross_session_guess(tmp_path):
    assert extract_candidates('I love it', 'user') == []
    assert extract_candidates('I love that', 'user') == []
    m = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    m.reconcile([{'role': 'user', 'content': 'I love it'}], 'new')
    m.reconcile([{'role': 'user', 'content': 'A device called Moonstone'}], 'other')
    m.reconcile([{'role': 'user', 'content': 'I like it'}], 'new')
    assert not m._get_store().fetch_states(scope='global', status='current')


@pytest.mark.parametrize('previous', [
    'We discussed Aster and Zero Bloom',
    'We were working on it',
    'I called the person about our project and another thing',
])
def test_ambiguous_previous_turn_is_not_a_fact(previous):
    assert resolve_preference_reference('I love it', previous) is None


def test_assistant_claims_are_never_personal_evidence(tmp_path):
    m = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    m.reconcile([{'role': 'assistant', 'content': 'You love a widget called Aster'},
                 {'role': 'user', 'content': 'I love it'}], 'same')
    assert m._get_store().fetch_states(scope='global', status='current') == []


def test_category_bounded_to_user_evidence(tmp_path):
    m = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    m.reconcile([{'role':'user','content':'A novel called Indigo River'},
                 {'role':'user','content':'I love it'}], 'reading')
    msg = m.verified_likes_history('what novels do i like?')
    assert 'Indigo River' in msg
    other = m.verified_likes_history('what sports do i like?')
    assert "couldn't verify" in other and 'Indigo River' not in other


def test_unclear_episode_subject_requests_clarification(tmp_path):
    m = ContextManager(db_path=tmp_path/'m.db', hybrid_retrieval=False)
    m.reconcile([{'role':'user','content':'I finished the River Garden redesign'}], 'prior')
    q = 'Aurora, remember when we worked on that activity?'
    assert parse_episode_question(q).needs_subject
    answer = m.verified_episode_history(q)
    assert 'Which activity' in answer
    assert 'River Garden' not in answer
    specific = m.verified_episode_history('What happened with the River Garden redesign?')
    assert 'River Garden redesign' in specific


class NoModel:
    def __init__(self): self.calls = 0
    def count_tokens(self, messages): return sum(len(m['content'])//4+1 for m in messages)
    def get_model_name(self): return 'test'
    def generate(self,*a,**kw):
        self.calls += 1
        return BrainResult.success('hallucinated a random preference','test')


def test_loop_retrieves_stored_preference_across_sessions(tmp_path):
    path = tmp_path/'m.db'
    m = ContextManager(db_path=path, hybrid_retrieval=False)
    m.reconcile([{'role':'user','content':'I am testing a gadget called Amber Bridge'},
                 {'role':'user','content':'I like it'}], 'older')
    brain = NoModel()
    loop = CognitiveLoop(Mind(ContextManager(db_path=path, hybrid_retrieval=False)),brain)
    answer = loop.run([{'role':'system','content':'Be accurate'},
                       {'role':'user','content':'What have I told you I like recently?'}],session_id='new')
    assert answer.success and 'Amber Bridge' in answer.content
    assert brain.calls == 0
