# %%
import numpy as np
import json
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import hist
from hist import Hist
import mplhep as hep
import matplotlib.pyplot as plt
import pandas as pd

# %%
class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    def process(self, events):
        dataset = events.metadata["dataset"]

        results={}
        results[dataset]={
            "count": len(events)
        }
            
        # define the histogram
        h = (
            Hist.new.StrCat([], growth=True, name="dataset", label="Primary dataset")
            .Reg(100, 60, 120, overflow=False, underflow=False, name="x", label = r"m$_{\gamma \gamma}$ [GeV]")
            .Weight()
        )

        h_eta_lead = (
            Hist.new
            .StrCat([], growth=True, name="dataset", label="Primary dataset")
            .Reg(100, -2.5, 2.5, overflow=False, underflow=False, name="x", label = r"m$_{\gamma \gamma}$ [GeV]")
            .Weight()
        )

        h_eta_sublead = (
            Hist.new
            .StrCat([], growth=True, name="dataset", label="Primary dataset")
            .Reg(100, -2.5, 2.5, overflow=False, underflow=False, name="x", label = r"m$_{\gamma \gamma}$ [GeV]")
            .Weight()
        )

        h_pt_lead = (
            Hist.new.StrCat([], growth=True, name="dataset", label="Primary dataset")
            .Reg(100, 30, 90, overflow=False, underflow=False, name="x", label = r"m$_{\gamma \gamma}$ [GeV]")
            .Weight()
            )
        
        
        h_pt_sublead = (
            Hist.new.StrCat([], growth=True, name="dataset", label="Primary dataset")
            .Reg(100, 30, 90, overflow=False, underflow=False, name="x", label = r"m$_{\gamma \gamma}$ [GeV]")
            .Weight()
            )

        # data_kind = "mc" if "GenPart" in ak.fields(events) else "data"

        electrons = events.Electron.mask[events.HLT.Ele32_WPTight_Gsf] 

        # get electrons

        # add selections
        count_number_of_electorn = ak.num(electrons, axis=1)
        
        # select electrons more than one
        electrons_masked = electrons.mask[count_number_of_electorn > 1]
        
        # tag tight cut electron based ID
        cutBased = electrons_masked.cutBased[:,0]
        electrons_masked = electrons_masked.mask[cutBased == 4]

        # select eta
        abs_eta = np.abs(electrons_masked.eta)
        electrons_masked = electrons_masked.mask[(abs_eta < 1.4442) | (abs_eta > 1.566) & (abs_eta < 2.1)]
        
        # select pt
        electrons_masked = electrons_masked.mask[
            (electrons_masked.pt[:,0] > 35)
            # (electrons_masked.pt[:,1] > 25)
        ]

        # save only the events passing the selections
        total_selection = ak.fill_none(
            ak.num(electrons_masked,axis=1) == 2,
            False
        )

        electrons_selected = electrons_masked[total_selection]

        # make sure the dielectron is e+ and e-
        charge = ak.sum(electrons_selected.charge, axis=1) == 0
        # make the dielectron pair combinations
        dielectron_pairs = ak.combinations(electrons_selected[charge], 2, fields=["lead", "sublead"])

        # dielectron four-momentum
        dielectrons = dielectron_pairs.lead+dielectron_pairs.sublead

        h.fill(dataset=dataset,x=dielectrons.mass[:,0])

        h_eta_lead.fill(dataset=dataset, x=dielectron_pairs.lead.eta[:,0])
        h_eta_sublead.fill(dataset=dataset, x=dielectron_pairs.sublead.eta[:,0])
        
        h_pt_lead.fill(dataset=dataset, x=dielectron_pairs.lead.pt[:,0])
        h_pt_sublead.fill(dataset=dataset, x=dielectron_pairs.sublead.pt[:,0])

        results["mass"] = h
        results["eta_lead"] = h_eta_lead
        results["eta_sublead"] = h_eta_sublead
        results["pt_lead"] = h_pt_lead
        results["pt_sublead"] = h_pt_sublead
        return results

    def postprocess(self, accumulant):
        pass

# %%
samplejson = "/eos/home-h/hhsieh/hsinyeh/TagAndProbe/configs/TnP_2018.json"
with open(samplejson) as f:
        sample_dict = json.load(f)

# %%
run = processor.Runner(
    # executor=processor.IterativeExecutor(),
    executor=processor.FuturesExecutor(workers=10), # user 4 cores
    schema=NanoAODSchema
)

results = run(
    sample_dict,
    treename="Events",
    processor_instance=MyProcessor(),
)



# %%
# results

# %%
# Data_hist= results["mass"][{"dataset":"Data"}]
# DY_hist= results["mass"][{"dataset":"DY"}]

# Data_values = Data_hist.values()
# DY_values = DY_hist.values()

# edges_Data = Data_hist.axes.edges
# edges_DY = DY_hist.axes.edges 

# print(DY_hist.values())


# %%
datasets_Data = [key for key in sample_dict if key.startswith("Data")]
datasets_DY = [key for key in sample_dict if key.startswith("DY")]
variables = ["mass","pt_lead","pt_sublead","eta_lead","eta_sublead"]

for variable in variables:
    Data_hist = sum(results[variable][{"dataset":dataset_Data}] for dataset_Data in datasets_Data)
    Data_values = Data_hist.values()
    edges_Data = Data_hist.axes.edges
    Data={
        'Data_bin_center':[(edges_Data[0][i] + edges_Data[0][i+1]) / 2 for i in range(len(edges_Data[0])-1)],
        'Data_count':Data_values,
    }
    df_Data = pd.DataFrame(data=Data)
    df_Data.to_parquet(f'parquet/Data_{variable}.parquet')



for variable in variables:
    DY_hist = sum(results[variable][{"dataset":dataset_DY}] for dataset_DY in datasets_DY)
    DY_values = DY_hist.values()
    edges_Data = Data_hist.axes.edges
    edges_DY = DY_hist.axes.edges 
    DY={
        'DY_bin_center':[(edges_DY[0][i] + edges_DY[0][i+1]) / 2 for i in range(len(edges_DY[0])-1)],
        'DY_count':DY_values,
    }
    df_DY = pd.DataFrame(data=DY)
    df_DY.to_parquet(f'parquet/DY_{variable}.parquet')
        







# %%
# df_Data

# %%
# hep.style.use(hep.style.CMS)

# f, ax = plt.subplots(figsize=(10,10))

# ax.set_ylabel("Count")
# results["mass"][{"dataset":"Data"}].plot(ax=ax,label="Data")
# results["mass"][{"dataset":"DY"}].plot(ax=ax,label="DY")

# hep.cms.label("Preliminary",loc=2,com=13.6)

# # ax.set_yscale("log")
# plt.legend()
# plt.savefig('pt1.png')


