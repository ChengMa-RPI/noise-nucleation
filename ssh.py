""" transfer files between local machine and server"""
import paramiko
import os 
degree = 4 
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000, 22500, 40000]
strength_set = [0.1, 0.5, 1, 5]
beta_fix_set = [4, 6, 6.97]

N_set = [10000] 
strength_set = [0.1, 0.5, 1]
strength_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
strength_set = [0.1, 0.2]

beta_fix_set = [4]
T = 10

client = paramiko.SSHClient()
client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

client.connect('ganxis3.nest.rpi.edu', username='mac6', password='woods*score&sister')

for N in N_set:
    for strength in strength_set:
        for beta_fix in beta_fix_set:

            server_des = f'/home/mac6/RPI/research/noise_transition/data/grid{degree}/size{N}/beta{beta_fix}/strength={strength}_T={T}/'
            local_des = f'/home/mac/RPI/research/noise_transition/data/grid{degree}/size{N}/beta{beta_fix}/strength={strength}_T={T}/'
            if not os.path.exists(local_des):
                os.makedirs(local_des)
            sftp = client.open_sftp()
            file_name = ['rho.csv', 'lifetime.csv']
            # file_name = ['rho.csv']
            # file_name = ['realization0.h5']
            for i in file_name:
                sftp.get(server_des + i, local_des + i)
            sftp.close()
