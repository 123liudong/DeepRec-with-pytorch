from examples.base_process import main


main(model_name='w&d', dataset_name='ML1m',
     dataset_path='E:\liudong\\feature_cross\借鉴的代码\pytorch-fm-master\data\\ml-1m\\ratings.dat',
     epoches=100, batch_size=2048,
     lr=0.001, device='cpu', save_path=None,
     model_params={
         'embed_size': 16,
         'hidden_nbs': [400, 400, 400, 1],
         'dropout': 0
     })