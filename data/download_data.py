import os

def downlaod(appliance, dataset):
    os.system(f"wget http://archive.ics.uci.edu/ml/machine-learning-databases/00442/{appliance}/{dataset} -O {appliance}_{dataset}")


os.system("wget http://archive.ics.uci.edu/ml/machine-learning-databases/00442/N_BaIoT_dataset_description_v1.txt")
os.system("wget http://archive.ics.uci.edu/ml/machine-learning-databases/00442/demonstrate_structure.csv")

appliances = ["Danmini_Doorbell",
              "Ecobee_Thermostat", 
              "Ennio_Doorbell",
              "Philips_B120N10_Baby_Monitor",
              "Provision_PT_737E_Security_Camera",
              "Provision_PT_838_Security_Camera",
              "Samsung_SNH_1011_N_Webcam",
              "SimpleHome_XCS7_1002_WHT_Security_Camera",
              "SimpleHome_XCS7_1003_WHT_Security_Camera"]

datasets = ["benign_traffic.csv",
            "gafgyt_attacks.rar",
            "mirai_attacks.rar"]

# Samsung_SNH_1011_N_Webcam and Ennio_Doorbell have no mirai_attacks.rar

for a in appliances:
    for d in datasets:
        downlaod(a, d)



