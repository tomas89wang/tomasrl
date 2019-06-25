from termios import *
import ctypes


__all__ = ['getch']


def cfmakeraw(flags):
    i_, o_, c_, l_, I_, O_, cc = flags
    i_ &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON)
    o_ &= ~(OPOST)
    c_ &= ~(CSIZE | PARENB)
    c_ |= CS8
    l_ &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN)
    return [i_, o_, c_, l_, I_, O_, cc[:]]


def getch():
    cdll = ctypes.CDLL('libc.so.6')
    flags = tcgetattr(0)
    nflags = cfmakeraw(flags)
    tcsetattr(0, TCSANOW, nflags)
    c = cdll.getchar()
    tcsetattr(0, TCSANOW, flags)
    return c


if __name__ == "__main__":
    c = getch()
    while c not in (3, 4, 27):
        print(type(c), c)
        c = getch()
