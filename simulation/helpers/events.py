from threading import Event


play_event = Event()
stop_event = Event()
pause_success_event = Event()


async def check_pause():
    await play_event.wait()
