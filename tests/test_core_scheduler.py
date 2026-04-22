from simweave.core.scheduler import EventQueue


def test_scheduled_events_fire_in_time_order():
    eq = EventQueue()
    fired: list[str] = []
    eq.schedule(5.0, lambda: fired.append("c"))
    eq.schedule(1.0, lambda: fired.append("a"))
    eq.schedule(3.0, lambda: fired.append("b"))
    for evt in eq.pop_due(now=5.0):
        evt.callback(*evt.args)
    assert fired == ["a", "b", "c"]


def test_peek_time_reports_next():
    eq = EventQueue()
    assert eq.peek_time() is None
    eq.schedule(10.0, lambda: None)
    eq.schedule(2.0, lambda: None)
    assert eq.peek_time() == 2.0


def test_cancelled_events_are_skipped():
    eq = EventQueue()
    evt1 = eq.schedule(
        1.0, lambda: (_ for _ in ()).throw(RuntimeError("should not fire"))
    )
    eq.schedule(2.0, lambda: None)
    eq.cancel(evt1)
    fired = list(eq.pop_due(5.0))
    assert len(fired) == 1
    assert fired[0].time == 2.0


def test_same_time_preserves_insertion_order():
    eq = EventQueue()
    fired = []
    eq.schedule(1.0, lambda: fired.append("first"))
    eq.schedule(1.0, lambda: fired.append("second"))
    eq.schedule(1.0, lambda: fired.append("third"))
    for evt in eq.pop_due(now=1.0):
        evt.callback()
    assert fired == ["first", "second", "third"]