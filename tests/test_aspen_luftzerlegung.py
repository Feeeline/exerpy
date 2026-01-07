from exerpy import ExergyAnalysis

model_path = r'C:\Users\Felin\Documents\Masterthesis\Code\Aspen\Masterthesis\Doppelkolonne_spez_Energie.bkp'

ean = ExergyAnalysis.from_aspen(model_path, chemExLib='Ahrendts', split_physical_exergy=False)

fuel = {
    "inputs": [
        "W1",  # elektrische Leistung Kompressor ZK1
        "W2",  # elektrische Leistung Kompressor ZK2
        "W3",  # elektrische Leistung Kompressor PK1
        ],
    "outputs": ["W4",  # elektrische Leistung Turbine T1
        ]
}

product = {
    "inputs": [],
    "outputs": ['S32']
}

loss = {
    "inputs": [],
    "outputs": ['S28','S25']
}

ean.analyse(E_F=fuel, E_P=product, E_L=loss)
ean.exergy_results()