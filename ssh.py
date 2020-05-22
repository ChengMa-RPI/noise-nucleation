""" transfer files between local machine and server"""
import paramiko
import os 

degree = 4 
N_set = [9, 16, 25, 36, 49, 64, 81, 100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000, 22500, 40000]
beta_fix_set = [4, 6, 6.97]
dynamics_all_set = ['mutual', 'harvest', 'eutrophication', 'vegetation']
c_all_set = [4, 2.6, 6, 2.7]
index = 0
dynamics = dynamics_all_set[index]
c_set = [c_all_set[index]]

N_set = [9, 16, 25, 36, 49, 64, 81, 100, 900, 2500]
sigma_set = [0.1]
R_set = [0, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 0.2]
sigma_set = [4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7, 1e-6, 5e-6]
sigma_set = [0.053]
R_set = [0.2]
N_set = [9, 100, 900, 2500, 10000]
N_set = [10000]

client = paramiko.SSHClient()
client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

client.connect('ganxis3.nest.rpi.edu', username='mac6', password='woods*score&sister')

def transfer_files(N_set, sigma_set, c_set, R_set, dynamics, directory, filenames):

    for N in N_set:
        for sigma in sigma_set:
            for c in c_set:
                for R in R_set:
                    if R == 0.2:
                        server_des = f'/home/mac6/RPI/research/noise_transition/data/' + dynamics + f'{degree}/size{N}/c{c}/strength={sigma}/'
                        local_des = f'/home/mac/RPI/research/noise_transition/data/' + dynamics + f'{degree}/size{N}/c{c}/strength={sigma}/'
                    else:
                        server_des = f'/home/mac6/RPI/research/noise_transition/data/' + dynamics + f'{degree}/size{N}/c{c}/strength={sigma}_R{R}/'
                        local_des = f'/home/mac/RPI/research/noise_transition/data/' + dynamics + f'{degree}/size{N}/c{c}/strength={sigma}_R{R}/'
                    if not os.path.exists(local_des):
                        os.makedirs(local_des)
                    sftp = client.open_sftp()
                    if '/' in directory:
                        if not os.path.exists(local_des + directory):
                            os.makedirs(local_des + directory)
                        filenames = sftp.listdir(server_des+directory) 
                    for i in filenames:
                        sftp.get(server_des + directory + i, local_des + directory +i)
                    sftp.close()

transfer_files(N_set, sigma_set, c_set, R_set, dynamics, '', ['lifetime.csv'])
# transfer_files(N_set, sigma_set, c_set, R_set, dynamics, '', ['nucleation.csv'])
# transfer_files(N_set, sigma_set, c_set, R_set, dynamics, 'ave/', [])
