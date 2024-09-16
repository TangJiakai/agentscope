from threading import Event


play_event = Event()
stop_event = Event()
kill_event = Event()
pause_success_event = Event()


def check_pause():
    play_event.wait()
