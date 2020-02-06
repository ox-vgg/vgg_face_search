__author__      = 'Ernesto Coto'
__copyright__   = 'April 2018'

import socket
import threading
import settings
import face_retrieval
import simplejson as json

# some hardcoded communication constants
TCP_TERMINATOR = "$$$"
SOCKET_TIMEOUT = 86400.00

class ThreadedServer(object):
    """
        Class implementing a basic socket server.
        Based on the code found at:
        https://stackoverflow.com/questions/23828264/how-to-make-a-simple-multithreaded-socket-server-in-python-that-remembers-client
    """

    def __init__(self, host, port):
        """
            Initializes the server
            Parameters:
                host: socket host name or IP
                port: port number at the host
        """
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))


    def listen(self):
        """
            Method that runs indefinitely waiting for connections
        """
        self.sock.listen(5)
        backend_instance = face_retrieval.FaceRetrieval()
        while True:
            try:
                client, address = self.sock.accept()
                client.settimeout(SOCKET_TIMEOUT)
                listening_thread = threading.Thread(target=self.listen_to_client, args=(client, backend_instance))
                listening_thread.start()
            except KeyboardInterrupt as e:
                print ('KeyboardInterrupt detected. Terminating Server !')
                backend_instance.worker_pool.terminate()
                backend_instance.worker_pool.join()
                break


    def listen_to_client(self, client, backend_instance):
        """
            Worker that serves an incoming connection
            Parameters:
                client: socket object usable to send and receive data on the connection
                backend_instance: instance of the face retrieval engine that will serve requests made over the connection
        """
        request = ""
        pid = threading.current_thread().ident
        while True:
            try:
                data = client.recv(1024)
                request += data.decode()
                if len(request) >= len(TCP_TERMINATOR):
                    if request[-len(TCP_TERMINATOR):] == TCP_TERMINATOR:
                        break
            except client.timeout:
                print ('Socket timeout')
                client.close()
            except Exception as e:
                print ('Exception in listenToClient: ' + str(e))
                client.close()
        try:
            request = request[:-len(TCP_TERMINATOR)]
            reply = backend_instance.serve_request(request, pid)
            reply = reply + TCP_TERMINATOR
            client.send(reply.encode())
            print ('Backend sent the reply')
        except Exception as e:
            print ('Exception in listenToClient: ' + str(e))
            pass

        client.close()


if __name__ == "__main__":
    # Start the server at the host and port specified in the settings file
    ThreadedServer(settings.HOST, settings.PORT).listen()
