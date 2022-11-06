""" 
transfer files between local machine and server
Changes need to be made:
Line 16 password; Line 26, 27 directory name

"""


import paramiko


client = paramiko.SSHClient()
client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
client.set_missing_host_key_policy(paramiko.RejectPolicy())

client.connect('jumpgate.scorec.rpi.edu', username='Jins4', password='your password')  # write your own password

def transfer_files(filenames):
    """
    transfer files from server to local machine.
    :filenames: the name of file to be transferred

    """

    "make sure that server_des and local_des have been created"
    server_des = 'directory in the server' 
    local_des =  'directory in the local machine'
    sftp = client.open_sftp()
    for i in filenames:
        sftp.get(server_des + i, local_des + i)
    sftp.close()

transfer_files(filenames)
