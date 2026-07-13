import pycuda.driver as cuda

cuda.init()

_device = cuda.Device(0)
_context = _device.make_context()
_context.pop()  # don't leave it active on whichever thread imports this


def push():
    _context.push()


def pop():
    _context.pop()