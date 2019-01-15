from sys import stdout
from time import sleep

# Write to the same line dynamically with this.
# Call finish_same_line() when your done with the line.
def write_same_line(x, sleep_time=0.01):
    stdout.write("\r%s" % str(x))
    stdout.flush()
    sleep(sleep_time)

def finish_same_line():
    stdout.write("\r  \r\n")