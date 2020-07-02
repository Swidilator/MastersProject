from datetime import timedelta, datetime


class RunTimer:

    __zero_time: timedelta = timedelta()

    def __init__(self, max_hours: float):
        self.__max_time_delta: timedelta = timedelta(hours=max_hours)
        self.end_time: datetime = datetime.now() + self.__max_time_delta
        self.start_time: datetime = datetime.now()
        self.max_interval_delta: timedelta = timedelta()
        self.last_interval_time: datetime = self.start_time
        self.reset_timer()

    def reset_timer(self) -> None:
        self.start_time = datetime.now()
        self.end_time = datetime.now() + self.__max_time_delta
        self.max_interval_delta: timedelta = timedelta()
        self.last_interval_time: datetime = self.start_time

    def update_and_predict_interval_security(self) -> bool:
        """
        Update the interval value and return security of next interval.

        :return: (bool): Security of next interval.
        """
        # Timer is disabled
        if self.__max_time_delta == RunTimer.__zero_time:
            return True

        now_time: datetime = datetime.now()
        self.max_interval_delta = max(
            self.max_interval_delta, now_time - self.last_interval_time
        )
        self.last_interval_time = now_time

        time_remaining: timedelta = (self.end_time - now_time)

        # Check if there is enough time, leaving enough time for saving
        if time_remaining > self.max_interval_delta + timedelta(minutes=1):
            print("Minutes remaining: {time}".format(time=time_remaining.seconds / 60))
            return True
        else:
            return False
