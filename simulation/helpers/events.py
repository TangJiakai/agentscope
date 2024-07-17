from threading import Event


play_event = Event()
stop_event = Event()


def check_pause():
    play_event.wait()
