# !pip install fastapi nest-asyncio websockets hypercorn azure-storage-blob azure-identity tqdm retry websocket_client psutil gputil tensorflow==2.6.4;


# from azure.storage.blob import BlobServiceClient
# from azure.identity import DefaultAzureCredential
import json
import os
import pickle
import random
import shutil
import time
from threading import Thread
import operator
import GPUtil as GPU
import numpy as np
import psutil
import requests
import torch
import torchvision.transforms as transforms
import websocket
from PIL import Image
from torch.autograd import Variable
import torchvision.datasets as dsets
from tqdm.auto import tqdm




################GLOBALS HERE########################

image_size = 32
label_dim = 10
G_input_dim = 100
G_output_dim = 1
D_input_dim = 1
D_output_dim = 1
num_filters = [512, 256, 128]

batch_size = 128
data_dir = 'MNIST_data/'
save_dir = 'MNIST_cDCGAN_results/'

transform = transforms.Compose([
    transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,), std=(0.5, ))
                                ])

mnist_data = dsets.MNIST(root=data_dir,
                        train=True,
                        transform=transform,
                        download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                        batch_size=batch_size,
                                        shuffle=True)





def to_np(x):
    return x.data.cpu().numpy()




def to_var(x):
    if torch.to.deviceis_available():
        x = x.to(device)
    return Variable(x)


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                        batch_size=batch_size,
                                        shuffle=True)


class Generator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Generator, self).__init__()

        # Hidden layers
        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                # For input
                input_deconv = torch.nn.ConvTranspose2d(input_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer1.add_module('input_deconv', input_deconv)

                # Initializer
                torch.nn.init.normal_(input_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_deconv.bias, 0.0)

                # Batch normal_ization
                self.hidden_layer1.add_module('input_bn', torch.nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.ReLU())

                # For label
                label_deconv = torch.nn.ConvTranspose2d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=1, padding=0)
                self.hidden_layer2.add_module('label_deconv', label_deconv)

                # Initializer
                torch.nn.init.normal_(label_deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_deconv.bias, 0.0)

                # Batch normal_ization
                self.hidden_layer2.add_module('label_bn', torch.nn.BatchNorm2d(int(num_filters[i]/2)))

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.ReLU())
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

                deconv_name = 'deconv' + str(i + 1)
                self.hidden_layer.add_module(deconv_name, deconv)

                # Initializer
                torch.nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(deconv.bias, 0.0)

                # Batch normal_ization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.ReLU())

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, label_dim, num_filters, output_dim):
        super(Discriminator, self).__init__()

        self.hidden_layer1 = torch.nn.Sequential()
        self.hidden_layer2 = torch.nn.Sequential()
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                # For input
                input_conv = torch.nn.Conv2d(input_dim, int(num_filters[i]/2), kernel_size=4, stride=2, padding=1)
                self.hidden_layer1.add_module('input_conv', input_conv)

                # Initializer
                torch.nn.init.normal_(input_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(input_conv.bias, 0.0)

                # Activation
                self.hidden_layer1.add_module('input_act', torch.nn.LeakyReLU(0.2))

                # For label
                label_conv = torch.nn.Conv2d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=2, padding=1)
                self.hidden_layer2.add_module('label_conv', label_conv)

                # Initializer
                torch.nn.init.normal_(label_conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(label_conv.bias, 0.0)

                # Activation
                self.hidden_layer2.add_module('label_act', torch.nn.LeakyReLU(0.2))
            else:
                conv = torch.nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

                conv_name = 'conv' + str(i + 1)
                self.hidden_layer.add_module(conv_name, conv)

                # Initializer
                torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
                torch.nn.init.constant_(conv.bias, 0.0)

                # Batch normal_ization
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

                # Activation
                act_name = 'act' + str(i + 1)
                self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Convolutional layer
        out = torch.nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal_(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant_(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, z, c):
        h1 = self.hidden_layer1(z)
        h2 = self.hidden_layer2(c)
        x = torch.cat([h1, h2], 1)
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out





def is_x_better_than_y(x,y,better_fn: operator = operator.lt):
    return better_fn(x,y)






def config_neural(args):
    generator_json = json.dumps(args['generator'])
    discriminator_json = json.dumps(args['discriminator'])




def get_generative():
    return Generator(G_input_dim, label_dim, num_filters, G_output_dim)




def get_discriminative():
    return  Discriminator(D_input_dim, label_dim, num_filters[::-1], D_output_dim)






def make_copy(filename):
    temp = filename.split('.')
    dest = temp[0]+'_copy.'+''.join(temp[1:])
    shutil.copyfile(filename, dest)
    return dest








# MNIST dataset


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)



##############WORKER PROCESS/FUNCTION########################


def boot_node(gpu_num):
    # RESOURCE MONITORING
    def get_bench():
        ram = psutil.virtual_memory()
        GPUs = GPU.getGPUs()
        gpu_bench = {}
        for idx, gpu in enumerate(GPUs):
            gpu_bench[f"GPU:{idx}"] = {}
            gpu_bench[f"GPU:{idx}"] = {"perc": round(
                gpu.memoryUtil*100, 2), "total": gpu.memoryTotal}


        per_cpu = psutil.cpu_percent(percpu=True)
        cpu_bench = {}
        for idx, usage in enumerate(per_cpu):
            cpu_bench[f'CPU_{idx}'] = usage


        return {
            'ram': {"perc": ram[2],
                    'total': round((ram[0]/1000000000), 2)},


            'cpu': cpu_bench,
            'gpu': gpu_bench
        }




    # CONFIG GLOBAL
    client_id = random.getrandbits(16)
    print("Client ID: ", client_id)
    base = '127.0.0.1:10000/' # 'evolve.westus3.cloudapp.azure.com:8000/'
    url = f"ws://{base}ws/{client_id}"
    master = websocket.WebSocket()
    MESSAGE = {"type": "status", "payload": "CONNECTING"}
    MESSAGE['bench'] = get_bench()
    ext = '.pt'
    device =  torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    print("Selected", device)


    ######################################################################
    # AZURE CONTAINER STUFF








    def config_container(args):
        return
        global blob_service_client, container_client, connect_str, container_name
        connect_str = args['connect_str']
        container_name = args['container_name']
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_client = blob_service_client.get_container_client(
            container=container_name)
        print(f"Blob Configured to {container_name}")




    def upload_file(path):
        return
        print(f"Starting upload of {path}")
        begin = time.time()
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=path)
        with open(path, 'rb') as file:
            blob_client.upload_blob(file, overwrite=True, connection_timeout=600)
        print(f"Upload Complete {path} in {(time.time()-begin):.2f}s")




    def ingress(url, name):
        return
        print("Starting ingress")
        begin = time.time()
        r = requests.get(url, allow_redirects=True)
        open(name, 'wb').write(r.content)
        print(f"Download from URL completed in {(time.time()-begin):.2f}s")
        upload_file(name)




    def download_file(path, redownload=True):
        return
        if (not redownload and os.path.exists(path)):
            print('File Exists')
            return
        print(f"Starting download of {path}")
        begin = time.time()
        with open(path, "wb") as download_file:
            download_file.write(container_client.download_blob(path).readall())
            print(f"Download of {path} complete in {(time.time()-begin):.2f}s")




    def replace_latest(src, dst):
        return
        download_file(src)
        print(f"Replacing {dst} with {src}")
        os.replace(src, dst)
        upload_file(dst)




    #############################################################################################




    #############################################################################################
    # NEURAL NETWORK CONFIG






    # Generator model
 




    def train(gen_name: str, disc_name: str, num_epochs: int, gen_lr: float, disc_lr: float, gen_beta1: float, disc_beta1: float, gen_beta2: float, disc_beta2: float, focus=None, v_freq=1, initial_gen_fitness=0,initial_disc_fitness=0, base = 0):
        initial_gen_fitness=100
        initial_disc_fitness=100
        MESSAGE['type'] = "status"
        MESSAGE["payload"] = f'PREPARING {gen_name} AND {disc_name}'


        criterion = torch.nn.BCELoss()
        step = 0
        D =  torch.jit.load(disc_name, map_location=device)
        G =  torch.jit.load(gen_name, map_location=device)
                   




    # Optimizers
        G_optimizer = torch.optim.Adam(G.parameters(), lr=gen_lr, betas=(gen_beta1, gen_beta2))
        D_optimizer = torch.optim.Adam(D.parameters(), lr=disc_lr, betas=(disc_beta1, disc_beta2))




        temp_noise = torch.randn(label_dim, G_input_dim)
        fixed_noise = temp_noise
        fixed_c = torch.zeros(label_dim, 1)
        for i in range(9):
            fixed_noise = torch.cat([fixed_noise, temp_noise], 0)
            temp = torch.ones(label_dim, 1) + i
            fixed_c = torch.cat([fixed_c, temp], 0)

        fixed_noise = fixed_noise.view(-1, G_input_dim, 1, 1)
        fixed_label = torch.zeros(G_input_dim, label_dim)
        fixed_label.scatter_(1, fixed_c.type(torch.LongTensor), 1)
        fixed_label = fixed_label.view(-1, label_dim, 1, 1)

        # label preprocess
        onehot = torch.zeros(label_dim, label_dim)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(label_dim, 1), 1).view(label_dim, label_dim, 1, 1)
        fill = torch.zeros([label_dim, label_dim, image_size, image_size])
        for i in range(label_dim):
            fill[i, i, :, :] = 1

        step = 0
        for epoch in tqdm(range(num_epochs)):
            D_losses = []
            G_losses = []

            if epoch == 5 or epoch == 10:
                G_optimizer.param_groups[0]['lr'] /= 2
                D_optimizer.param_groups[0]['lr'] /= 2

            # minibatch training
            for i, (images, labels) in enumerate((pbar:=tqdm(data_loader))):

                # image data
        
                mini_batch = images.size()[0]
                x_ = Variable(images.to(device))

                # labels
                y_real_ = Variable(torch.ones(mini_batch).to(device))
                y_fake_ = Variable(torch.zeros(mini_batch).to(device))
                c_fill_ = Variable(fill[labels].to(device))

                # Train discriminator with real data
                D_real_decision = D(x_, c_fill_).squeeze()
                D_real_loss = criterion(D_real_decision, y_real_)

                # Train discriminator with fake data
                z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
                z_ = Variable(z_.to(device))

                c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
                c_onehot_ = Variable(onehot[c_].to(device))
                gen_image = G(z_, c_onehot_)

                c_fill_ = Variable(fill[c_].to(device))
                D_fake_decision = D(gen_image, c_fill_).squeeze()
                D_fake_loss = criterion(D_fake_decision, y_fake_)

                # Back propagation
                D_loss = D_real_loss + D_fake_loss
                D.zero_grad()
                D_loss.backward()
                D_optimizer.step()

                # Train generator
                z_ = torch.randn(mini_batch, G_input_dim).view(-1, G_input_dim, 1, 1)
                z_ = Variable(z_.to(device))

                c_ = (torch.rand(mini_batch, 1) * label_dim).type(torch.LongTensor).squeeze()
                c_onehot_ = Variable(onehot[c_].to(device))
                gen_image = G(z_, c_onehot_)

                c_fill_ = Variable(fill[c_].to(device))
                D_fake_decision = D(gen_image, c_fill_).squeeze()
                G_loss = criterion(D_fake_decision, y_real_)

                # Back propagation
                G.zero_grad()
                G_loss.backward()
                G_optimizer.step()

                # loss values
                D_losses.append(D_loss.item())
                G_losses.append(G_loss.item())

                pbar.set_description('%s %s %s Epoch [%d/%d], Step [%d/%d], D_loss: %f, G_loss: %f'
                    % (device, gen_name, disc_name, epoch+1, num_epochs, i+1, len(data_loader), D_loss.item(), G_loss.item()))


                step += 1

            D_avg_loss = torch.mean(torch.FloatTensor(D_losses)).item()
            G_avg_loss = torch.mean(torch.FloatTensor(G_losses)).item()




            payload = {}
            if (focus != 'generator' and is_x_better_than_y(D_avg_loss,initial_disc_fitness, operator.lt)):
                print(
                    f"{disc_name} Fitness improved from {initial_disc_fitness} to {D_avg_loss}")
                initial_disc_fitness = D_avg_loss
                payload[f"{disc_name.split('.')[0]}"] ={"loss":f"{D_avg_loss}"}
                model_scripted = torch.jit.script(D)
                model_scripted.save(disc_name)
                upload_file(disc_name)
            else:
                print(f"{disc_name} ignored")


            if (focus != 'discriminator' and is_x_better_than_y(G_avg_loss, initial_gen_fitness, operator.lt)):
                print(f"{gen_name} Fitness improved from {initial_gen_fitness} to {G_avg_loss}")
                initial_gen_fitness = G_avg_loss
                payload[f"{gen_name.split('.')[0]}"] = {"loss": f"{G_avg_loss}"}
                model_scripted = torch.jit.script(G)
                model_scripted.save(gen_name)
                upload_file(gen_name)
            else:
                print(f"{gen_name} ignored")


            packet = {
                "type": "fitness",
                "payload": payload
            }


            send_update(packet)


            time.sleep(0.5)






    #############################################################################################
    # WEBSOCKET CONNECTION CONFIG


    def connect_to_master():


        while True:
            try:
                master.connect(url)
                print(f"Worker Connected to {url}")
                break
            except ConnectionRefusedError:
                print("Connection Refused")
                time.sleep(3)




    def listen():
        while True:


            try:
                data = master.recv()
                parse_command(data)
                time.sleep(0.1)
            except (ConnectionResetError, BrokenPipeError, websocket.WebSocketConnectionClosedException):
                print("Connection Reset, trying to reconnect")
                try:
                    master.connect(url)
                    print(f"Worker Connected to {url}")
                except Exception:
                    print("Connection Refused, , retrying in 5 seconds")
                    time.sleep(5)




    def send(data):
        try:
            master.send(str(data))
        except (ConnectionResetError, BrokenPipeError, websocket.WebSocketConnectionClosedException):
            print("Connection Reset, trying to reconnect")
            try:
                master.connect(url)
                print(f"Worker Connected to {url}")
            except Exception:
                print("Connection Refused, retrying in 5 seconds")
                time.sleep(5)




    def parse_command(data):
        try:
            if type(data) == dict:
                msg = data
            elif type(data) == str:
                try:
                    msg = json.loads(data)
                except:
                    import ast
                    msg=ast.literal_eval(data)  
        except ValueError:
            print("Could not parse:",data, " of type ", type(data))  # invalid json
        else:
            if (msg['type'] == 'container_cred'):
                args = msg['args']
                config_container(args)


            elif (msg['type'] == 'config_neural'):
                args = msg['args']
                print("Configuring Neural architectures ")
                config_neural(args)
                MESSAGE["type"] = "status",
                MESSAGE["payload"] = "Configuring NN"


            elif (msg['type'] == 'create'):
                args = msg
                model = args['create']
                name = args['name']
                MESSAGE["type"] = "status",
                MESSAGE["payload"] = f"CREATING {name}"


                if (model == 'generator'):
                    gen = get_generative()
                    model_scripted = torch.jit.script(gen)
                    del gen
                    model_scripted.save(f"{name}{ext}")
                    upload_file(f"{name}{ext}")
                    print(f"Created and uploaded {name}{ext}")
                elif (model == 'discriminator'):
                    disc = get_discriminative()
                    model_scripted = torch.jit.script(disc)
                    del disc
                    model_scripted.save(f"{name}{ext}")
                    upload_file(f"{name}{ext}")
                    print(f"Created and uploaded {name}{ext}")




                packet = {
                "type": "finished",
                "command": msg['id']
                }


                send_update(packet)


                time.sleep(1)
                MESSAGE["type"] = "status",
                MESSAGE["payload"] = "READY"


            elif (msg['type'] == 'train'):


                args = msg


                gen_name = args['generator']
                disc_name = args['discriminator']
                dataset = args['dataset']
                gen_lr = float(args['gen_lr'])
                disc_lr = float(args['disc_lr'])
                gen_beta1 = float(args['gen_beta1'])
                disc_beta1 = float(args['disc_beta1'])
                gen_beta2 = float(args['gen_beta2'])
                disc_beta2 = float(args['disc_beta2'])
                epochs = int(args['epochs'])
                mode = args['mode']
                MESSAGE["type"] = "status"
                MESSAGE["payload"] = "TRAINING STARTED"
                focus = None
                download_file(gen_name)
                download_file(disc_name)
                if (mode == 'copy'):
                    focus = args['focus']
                    gen_name = make_copy(gen_name)
                    disc_name = make_copy(disc_name)
                train(gen_name, disc_name, epochs, gen_lr, disc_lr,
                    gen_beta1, disc_beta1, gen_beta2, disc_beta2, focus=focus)




                packet = {
                "type": "finished",
                "command": msg['id']
                }
                print("Train finished")


                send_update(packet)
                time.sleep(1)
                MESSAGE["type"] = "status"
                MESSAGE["payload"] = "READY"


            elif (msg['type'] == 'replace'):
                args = msg
                src = args['source']+f'{ext}'
                dst = args['destination']+f'{ext}'
                replace_latest(src, dst)


                packet = {
                "type": "finished",
                "command": msg['id']
                }


                send_update(packet)
                time.sleep(1)
                MESSAGE["type"] = "status"
                MESSAGE["payload"] = "READY"


            elif (msg['type'] == 'download_blob'):
                filename = msg['filename']
                redownload = msg['redownload'] == 'True'
                download_file(filename, redownload)


            elif (msg['type'] == 'call'):
                if (True):


                    MESSAGE["type"] = "status"
                    MESSAGE["payload"] = "READY"
                else:
                    print("Data load failure")
                    MESSAGE["type"] = "status"
                    MESSAGE["payload"] = "DATASET LOAD FAILED"


            elif (msg['type'] == 'exit'):
                exit()


            else:
                print("Unkown command")




    def send_update(packet: dict):
        url = 'http://'+base+'update/'+str(client_id)
        x = requests.post(url, json=packet)
        # print(x.content)




    def update_bench():


        while True:
            MESSAGE['bench'] = get_bench()
            time.sleep(0.4)




    #############################################################################################




    print("Worker ready to connect")
    connect_to_master()


    listener = Thread(target=listen, args=())
    listener.start()


    bench = Thread(target=update_bench, args=())
    bench.start()


    while (True):
        send(json.dumps(MESSAGE))
        time.sleep(0.4)


from multiprocessing import Process


if __name__ == '__main__':
    processes = []
    for device in range(torch.cuda.device_count()):
        processes.append(Process(target=boot_node, args=(device,)))
        processes[device].start()
        
