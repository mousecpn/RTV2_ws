"""
Communication protocol implemented with ZeroMQ + MessagePack.
"""
import logging
from typing import Any, Callable, List, Tuple, Union

import msgpack
import zmq


def Address(host: str, port: int):
    """Create tcp address."""
    return "tcp://{}:{}".format(host, port)


def SetTopic(sock: zmq.Socket, topic: Union[List[Union[str, bytes]], str, bytes]):
    """Set one or more topics on the given socket.."""
    if not isinstance(topic, list):
        topic = [topic]
    for t in topic:
        if not isinstance(t, bytes):
            t = t.encode()  # Topics must be bytes.
        sock.setsockopt(zmq.SUBSCRIBE, t)


def Poll(sock: zmq.Socket, timeout: Union[int, None] = None) -> None:
    """Poll socket and raise OSError in case of timeout (in mSec)."""
    if timeout is not None:
        if not sock.poll(timeout, zmq.POLLIN):
            raise OSError("receive timeout")


def SendMessage(sock: zmq.Socket, identifier: str, message: Any = None) -> None:
    """Send multi-part message.

    Sends 2-frame message with identifier (as bytes) and the message (serialized using msgpack).
    Counterpart of ReceiveMessage."""
    logging.debug("%08d send [%s]: %s", sock.underlying, identifier, message)
    sock.send_multipart([identifier.encode(), msgpack.packb(message)])


def ReceiveMessage(sock: zmq.Socket, timeout: Union[int, None] = None, flags: int = 0) -> Tuple[str, Any]:
    """Receive multi-part message.

    Expects 2-frame message with identifier (as bytes) and the message (deserialized using msgpack),
    polls during timeout first then decodes and deserializes and returns both parts.
    Counterpart of SendMessage."""
    Poll(sock, timeout=timeout)
    identifier, payload = sock.recv_multipart(flags)
    identifier = identifier.decode()
    # Use raw=False to get string not bytes.
    message = msgpack.unpackb(payload, raw=False)
    logging.debug("%08d recv [%s]: %s", sock.underlying, identifier, message)
    return identifier, message


def Flush(
    sock: zmq.Socket,
    until: Union[None, str] = None,
    timeout: Union[None, int] = 500,
    iterations: Union[None, int] = 10,
    recv: Callable[[zmq.Socket, int], Tuple[str, str]] = ReceiveMessage,
    raising: bool = True,
) -> Union[None, Any]:
    """Keep reading messages until timout expires or until message with id 'until' is read.

    Returns the message if any, else None.
    If until is None raises OSError in case of timeout unless raising is False.
    This works in iterations of timeout, because in some occasions it seems that works
    better, whereas using one big timeout just gets no message through.
    Once a message is received, an immediate attempt to read more (without timeout) follows,
    to flush as fast as possible.
    If timeout is None blocks until messages are received. Likewise if iterations=None,
    blocks until the wanted message is received.
    """
    logging.debug("%08d flush", sock.underlying)
    if until is None and iterations is None:
        raise ValueError("flush would lead to infinite loop")
    numberOfMessages = 0

    def Iteration():
        nonlocal numberOfMessages
        usedTimeout = timeout
        try:
            while True:
                id, message = recv(sock, usedTimeout)
                # We'll try again immediately.
                usedTimeout = 0
                numberOfMessages += 1
                if until is not None and id == until:
                    logging.debug("%08d flush %d messages", sock.underlying, numberOfMessages)
                    return message
        except OSError:
            logging.debug("%08d flush [timeout]", sock.underlying)

    if iterations is None:
        while True:
            if result := Iteration() is not None:
                return result
    else:
        for _ in range(iterations):
            if result := Iteration() is not None:
                return result

    logging.debug("%08d flush %d messages", sock.underlying, numberOfMessages)
    if until is not None:
        raise OSError(f"timeout waiting for '{until}'")
    elif raising:
        raise OSError("flush timeout")
