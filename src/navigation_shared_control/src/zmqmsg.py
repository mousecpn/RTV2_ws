import logging
import sys
import time

import zmq
import tns.zmq

# handler = logging.StreamHandler(sys.stdout)
# handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
# logging.root.addHandler(handler)
# logging.root.setLevel(logging.DEBUG)

# context = zmq.Context()
# subscriber = context.socket(zmq.REP)
# publisher = context.socket(zmq.REQ)
# subscriber.bind(tns.zmq.Address("*", 33458))
# publisher.bind(tns.zmq.Address("*", 33456))


def SendMessage(sock: zmq.Socket, identifier: str, message=None, timeout: int = 5000) -> None:
    """Send message, for REP-REQ.

    Sends 2-frame message with identifier (as bytes) and the message (serialized using msgpack),
    then waits for the reply string "ok" or raises for timeout or when another reply is received..
    Counterpart of ReceiveMessage."""
    tns.zmq.SendMessage(sock, identifier, message)
    tns.zmq.Poll(sock, timeout=timeout)
    reply = sock.recv().decode()
    logging.debug("%08d got reply: %s", sock.underlying, reply)
    if reply != "ok":
        raise OSError(f"expected 'ok' but got '{reply}")


def ReceiveMessage(sock: zmq.Socket, timeout=None):
    """Receive message, for REP-REQ.

    Expects 2-frame message with identifier (as bytes) and the message (serialized using msgpack),
    decodes and deserializes then send the reply string "ok".
    Counterpart of SendMessage."""
    identifier, message = tns.zmq.ReceiveMessage(sock, timeout)
    logging.debug("%08d send [ok]", sock.underlying)
    sock.send_string("ok")
    return identifier, message