#DayDayUpQ2.py
dayfactor = 0.01
dayupp = pow(1 + dayfactor, 365)
daydownw = pow(1 - dayfactor, 365)
print("向上：{:.2f}，向下：{:.2f}".format(dayupp, daydownw))
