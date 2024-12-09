from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.pipeline import perf_data

params={
 'collection_name': 'QMugs_DFT_FRCE2',
 'dataset_key': "/home/thangavelucs/05152023_HPOBatchScriptTest/QMugs_curatedDFT.csv",
 'datastore': 'False',
 "featurizer": 'ecfp', 
 'id_col': 'compound_id',
 'model_type': 'NN',
 'prediction_type': 'regression',
 'previously_split': 'True',
 'split_uuid': '02e0c361-4b5b-499e-9a74-39faac54feb3',
 'rerun': 'False',
 'response_cols': ['VALUE_NUM_mean'],
 'max_epochs': '40',
 'result_dir': "/home/thangavelucs/05152023_HPOBatchScriptTest/result_dir2",
 'save_results': 'False',
 'search_type': 'user_specified',
 'smiles_col': 'rdkit_smiles',
 'split_only': 'False',
 'splitter': 'scaffold',
 'transformers': 'True',
 'uncertainty': 'True',
 'verbose': 'True'}

layer_dropout = [
                 ('64,16','0.0,0.0'),
                 ('64,16','0.3,0.3'),
                 ('128,32','0.0,0.0'),
                 ('128,32','0.3,0.3'),
                 ('256,64','0.0,0.0'),
                 ('256,64','0.3,0.3'),
                 ('512,128','0.0,0.0'),
                 ('512,128','0.3,0.3'),
                 ('256,64,16','0.0,0.0,0.0'),
                 ('256,64,16','0.3,0.3,0.3'),
                 ('512,256,32','0.0,0.0,0.0'),
                 ('512,256,32','0.3,0.3,0.3'),
                 ('512,256,128','0.0,0.0,0.0'),
                 ('512,256,128','0.3,0.3,0.3'),
                 ('512,128,128','0.3,0.3,0.3'),
                 ('512,128,128','0.0,0.0,0.0'),
                 ('512,256,128,64','0.0,0.0,0.0,0.0'),
                 ('512,256,128,64','0.3,0.3,0.3,0.3')
]

lr_choice = [.00001,.00005,.0001,.0005,.001,.01]


for layers,dropouts in layer_dropout:
    for learning_rate in lr_choice:
        params["layer_sizes"] = layers
        params["dropouts"] = dropouts
        params["learning_rate"] = learning_rate
        tp = parse.wrapper(params)
        pl = mp.ModelPipeline(tp)
        pl.train_model()
        pred_data = pl.model_wrapper.get_perf_data(subset="valid", epoch_label="best")
        pred_results = pred_data.get_prediction_results()
        print(f"layers: {layers}, dropouts: {dropouts}, learning rate: {learning_rate}, valid_r2: {pred_results['r2_score']}\n")