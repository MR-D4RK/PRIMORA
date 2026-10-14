import sys

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.specialists import (
    Specialist,
    CalculateSpecialist,
    RecallSpecialist,
    ResearchSpecialist,
    ResearchReadSpecialist,
    ConsultSpecialist,
    LearnSpecialist,
    UseToolSpecialist,
)


def test_all_specialists_exported():
    assert issubclass(CalculateSpecialist, Specialist)
    assert issubclass(RecallSpecialist, Specialist)
    assert issubclass(ResearchSpecialist, Specialist)
    assert issubclass(ResearchReadSpecialist, Specialist)
    assert issubclass(ConsultSpecialist, Specialist)
    assert issubclass(LearnSpecialist, Specialist)
    assert issubclass(UseToolSpecialist, Specialist)


def test_names_match_capability_actions():
    assert CalculateSpecialist.name == "calculate"
    assert RecallSpecialist.name == "recall"
    assert ResearchSpecialist.name == "research"
    assert ResearchReadSpecialist.name == "research_read"
    assert ConsultSpecialist.name == "consult_specialist"
    assert LearnSpecialist.name == "learn"
    assert UseToolSpecialist.name == "use_tool"
