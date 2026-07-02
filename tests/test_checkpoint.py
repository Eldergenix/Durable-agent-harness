import pytest

from dah.checkpoint import Checkpointer, DictSnapshotMixin, Snapshotable


class Counter(DictSnapshotMixin):
    def __init__(self):
        self.value = 0
        self.log = []


def test_dict_snapshot_mixin_is_snapshotable():
    assert isinstance(Counter(), Snapshotable)


def test_rollback_restores_state():
    c = Counter()
    cp = Checkpointer()
    cp.register("c", c)
    c.value = 1
    cp.commit("one")
    c.value = 2
    c.log.append("mutated")
    cp.rollback()
    assert c.value == 1
    assert c.log == []  # deep restore, not aliased


def test_rollback_is_deep_and_independent():
    c = Counter()
    cp = Checkpointer()
    cp.register("c", c)
    c.log.append("a")
    cp.commit("with-a")
    c.log.append("b")
    cp.rollback()
    # mutating after rollback must not corrupt the stored snapshot
    c.log.append("c")
    cp.rollback()
    assert c.log == ["a"]


def test_rollback_to_specific_id_discards_later():
    c = Counter()
    cp = Checkpointer()
    cp.register("c", c)
    c.value = 1
    first = cp.commit("first")
    c.value = 2
    cp.commit("second")
    cp.rollback(to=first.id)
    assert c.value == 1
    assert cp.depth == 1  # "second" discarded


def test_pop_does_not_restore():
    c = Counter()
    cp = Checkpointer()
    cp.register("c", c)
    cp.commit("x")
    c.value = 9
    cp.pop()
    assert c.value == 9  # pop keeps the live state
    assert cp.depth == 0


def test_rollback_without_checkpoint_errors():
    cp = Checkpointer()
    with pytest.raises(RuntimeError):
        cp.rollback()


def test_register_rejects_non_snapshotable():
    cp = Checkpointer()
    with pytest.raises(TypeError):
        cp.register("bad", object())
