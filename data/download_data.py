import os
import pandas as pd

def run(s): return os.system(s)
def cd(s): return os.chdir(s)

def downlaod(appliance, dataset):
    return run(f"wget http://archive.ics.uci.edu/ml/machine-learning-databases/00442/{appliance}/{dataset}")


#run("wget http://archive.ics.uci.edu/ml/machine-learning-databases/00442/N_BaIoT_dataset_description_v1.txt")
#run("wget http://archive.ics.uci.edu/ml/machine-learning-databases/00442/demonstrate_structure.csv")

appliances = ["Danmini_Doorbell",
              "Ecobee_Thermostat", 
              "Philips_B120N10_Baby_Monitor",
              "Provision_PT_737E_Security_Camera",
              "Provision_PT_838_Security_Camera",
              "SimpleHome_XCS7_1002_WHT_Security_Camera",
              "SimpleHome_XCS7_1003_WHT_Security_Camera",
              "Samsung_SNH_1011_N_Webcam",
              "Ennio_Doorbell"]

datasets = ["benign_traffic.csv",
            "gafgyt_attacks.rar",
            "mirai_attacks.rar"]

# Samsung_SNH_1011_N_Webcam and Ennio_Doorbell have no mirai_attacks.rar so for now are ignored

run("mkdir temp")
cd("temp")

for app in appliances[:7]:
    for d in datasets:
        print(downlaod(app, d))

    df = pd.read_csv("benign_traffic.csv")
    df["type"] = "benign"
    run("rm benign_traffic.csv")

    all_dataframes = [df]

    run("unrar e gafgyt_attacks.rar")
    run("rm gafgyt_attacks.rar")
    for name in ["combo", "junk", "scan", "tcp", "udp"]:
        df = pd.read_csv(name+".csv")
        df["type"] = "gafgyt_" + name
        all_dataframes.append(df)
        run("rm "+name+".csv")

    run("unrar e mirai_attacks.rar")
    run("rm mirai_attacks.rar")
    for name in ["ack", "scan", "syn", "udp", "udpplain"]:
        df = pd.read_csv(name+".csv")
        df["type"] = "mirai_" + name
        all_dataframes.append(df)
        run("rm "+name+".csv")

    dataframe = pd.concat(all_dataframes)
    # shuffel dataframe maybe?
    dataframe.to_csv(f"../{app}_complet.csv", index=False)

cd("..")
run("rm -rf temp")
