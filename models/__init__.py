'''
init script for models: 
    It contains a Factory class, that encapsulate the instantiation of the different imputation models
'''

import os

class Factory:
    def __init__(self, model: str, seq_len: int = 256, n_features: int = 4,
                batch_size:int = 32, epochs: int = 50, patience: int = 15,
                optimizer = None, device: str = 'cpu', output_dir: str = '.',
                learning_rate: float = 1e-3) -> None:
        '''
        Factory class. It is used to colect and to setup the instantion function of the different models.
        Unless specified, the hyperparameter were empirically selected.
        
        @param model: the name of the selected model
        @param seq_len: the length (number of samples) of the training/validation sequence 
        @param n_features: the number of features of each sequence
        @param batch_size: the size of a training/validation batch
        @param epochs: the number of training epochs                                                              # Not used by shallow/classical methods
        @param patience: the number of patience epochs. 
                         Aborts training if validation score do not improve in this number of epochs              # Not used by shallow/classical methods
        @param optimizer: an optimizer object                                                                     # Not used by shallow/classical methods
        @param device: the device used for training. (cpu or cuda usually)                                        # Not used by shallow/classical methods
        @param output_dir: the path where model weights and trainig logs are saved
        @param learning_rate: learning rate for repository-native neural models
        '''
        
        
        assert model in ['locf', 'mean', # classical (mean not implemented)
                         'rf', 'qrf', 'quantilerf', 'xgboost', 'svm', # shallow/ensemble methods
                         'saits', 'transformer', # attenttion time-series (pypots implementation)
                         'brits', 'mrnn', # rnn time-series (pypots implementation)
                         'unet', # cnn based method (monai backbone implementation)
                         'ae', 'bayesnn', 'bayessnn', # neural methods implemented in this repository
                        ]
        
        self.model = model
        self.seq_len = seq_len
        self.n_features = n_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.device = device
        self.output_dir = output_dir
    
    def instantiate(self):
        model = getattr(self, f"instantiate_{self.model.upper()}")(self.seq_len, self.n_features, self.batch_size, self.epochs, self.patience, self.optimizer, self.device, self.output_dir)
        if self.model in {'bayesnn', 'bayessnn'}:
            model.learning_rate = self.learning_rate
        return model
    
    @staticmethod
    def instantiate_SAITS(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from pypots.imputation import SAITS
        
        saits = SAITS(n_steps = seq_len,
                      n_features = n_features,
                      n_layers = 2,
                      d_model = 256,
                      d_inner = 128,
                      n_heads = 4,
                      d_k = 64,
                      d_v = 64,
                      dropout = 0.1,
                      attn_dropout = 0.1,
                      diagonal_attention_mask = True,  # otherwise the original self-attention mechanism will be applied
                      ORT_weight=1,  # you can adjust the weight values of arguments ORT_weight and MIT_weight to make the SAITS model focus more on one task. Usually you can just leave them to the default values, i.e. 1.
                      MIT_weight=1,
                      batch_size = batch_size,
                      epochs = epochs,
                      patience = patience,
                      optimizer = optimizer,
                      num_workers = 0,
                      device = device, 
                      saving_path = os.path.join(output_dir, 'saits'), # set the path for saving tensorboard and trained model files
                      model_saving_strategy = 'best', # only save the best model after training finished.
                     )
    
        return saits
    
    @staticmethod
    def instantiate_TRANSFORMER(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from pypots.imputation import Transformer
        
        transformer = Transformer(n_steps = seq_len,
                                  n_features = n_features,
                                  n_layers = 6,
                                  d_model = 128,
                                  d_inner = 64,
                                  n_heads = 4,
                                  d_k = 64,
                                  d_v = 64,
                                  dropout = 0.1,
                                  attn_dropout = 0,
                                  ORT_weight = 1,
                                  MIT_weight = 1,
                                  batch_size = batch_size,
                                  epochs = epochs,
                                  patience = patience,
                                  optimizer = optimizer,
                                  num_workers = 0,
                                  device = device,
                                  saving_path = os.path.join(output_dir, 'transformer'),
                                  model_saving_strategy = 'best',
                                 )
        return transformer
    
    @staticmethod
    def instantiate_BRITS(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from pypots.imputation import BRITS
        
        brits = BRITS(n_steps = seq_len,
                      n_features = n_features,
                      rnn_hidden_size = 128,
                      batch_size = batch_size,
                      epochs = epochs,
                      patience = patience,
                      optimizer = optimizer,
                      num_workers = 0,
                      device = device,
                      saving_path = os.path.join(output_dir, 'brits'), 
                      model_saving_strategy = 'best',
                     )
        
        return brits
    
    @staticmethod
    def instantiate_MRNN(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from pypots.imputation import MRNN
        
        mrnn = MRNN(n_steps = seq_len,
                    n_features = n_features,
                    rnn_hidden_size = 128,
                    batch_size = batch_size,
                    epochs = epochs,
                    patience = patience,
                    optimizer = optimizer,
                    num_workers = 0,
                    device = device,
                    saving_path = os.path.join(output_dir, 'mrnn'),
                    model_saving_strategy = 'best',
                   )
        
        return mrnn
    
    @staticmethod
    def instantiate_LOCF(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from pypots.imputation import LOCF
        
        locf = LOCF(
                    first_step_imputation="zero"  # fill leading missing values, which cannot use LOCF
                   )
        
        return locf
    
    @staticmethod
    def instantiate_UNET(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from .unet import UNet
        
        unet = UNet(n_features = n_features,
                    spatial_dims = 1,
                    channels = tuple([2**(i+5) for i in range(5)]),
                    strides= (2,2,2,2),
                    num_res_units = 1,
                    batch_size = batch_size,
                    epochs = epochs,
                    patience = patience,
                    optimizer = optimizer,
                    num_workers = 0,
                    device = device,
                    saving_path = os.path.join(output_dir, 'unet'),
                    model_saving_strategy = 'best',
                   )

        return unet
    
    @staticmethod
    def instantiate_AE(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from .autoencoder import AENet
        
        ae = AENet(n_features=n_features,
                   input_size=seq_len,
                   enc_layers=[128, 64],
                   enc_activation='relu',
                   dec_layers=None,
                   dec_activation=None,
                   mirrored=True,
                   latent_dim=32,
                   batch_size = batch_size,
                   epochs= epochs,
                   patience = patience,
                   optimizer = optimizer,
                   num_workers = 0,
                   device = device,
                   saving_path = os.path.join(output_dir, 'ae'),
                   model_saving_strategy = "best"
                  )

        return ae
    
    @staticmethod
    def instantiate_RF(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from .shallow import RF
        # Hyperparameters from: Synthetic geochemical well logs generation using ensemble machine learning techniques for the Brazilian pre-salt reservoirs
        # de Oliveira, Lucas Abreu Blanes and de Carvalho Carneiro, Cleyton, [2021]
        rf = RF(n_features,
                device=None,
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2,
               )
    
        return rf

    @staticmethod
    def instantiate_QRF(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from .quantilerf import QuantileRandomForest

        return QuantileRandomForest(
            n_features,
            # CPU-friendly baseline. QRF trains one forest per feature, so
            # unrestricted 200-tree forests are prohibitively expensive on
            # the full set of depth samples.
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            min_samples_split=10,
            max_samples=0.5,
            n_jobs=-1,
            random_state=17076,
        )

    @staticmethod
    def instantiate_BAYESNN(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from .bayessnn import BayesianNNImputer

        return BayesianNNImputer(
            num_models=n_features,
            # BayesNN flattens sequences into individual depth points. The
            # global batch size of 32 is intended for full sequences and is
            # prohibitively small here.
            batch_size=max(batch_size, 2048),
            epochs=epochs,
            patience=patience,
            device=device,
            learning_rate=1e-3,
            hidden_size=64,
            dropout=0.15,
            mc_samples=30,
            min_delta=1e-4,
        )

    @staticmethod
    def instantiate_QUANTILERF(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        return Factory.instantiate_QRF(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir)

    @staticmethod
    def instantiate_BAYESSNN(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        return Factory.instantiate_BAYESNN(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir)
    
    @staticmethod
    def instantiate_XGBOOST(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from .shallow import XGBOOST
        # Hyperparameters from: Synthetic geochemical well logs generation using ensemble machine learning techniques for the Brazilian pre-salt reservoirs
        # de Oliveira, Lucas Abreu Blanes and de Carvalho Carneiro, Cleyton, [2021]
        xgboost = XGBOOST(n_features,
                          device=None,
                          n_estimators=26,
                          min_child_weight=6.,
                          gamma=0.,
                          subsample=0.8,
                          colsample_bytree=1,
                          reg_alpha=0.,
                          learning_rate=0.1
                         )
    
        return xgboost
    
    @staticmethod
    def instantiate_SVM(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        from .shallow import SVM
        # Hyperparameters from: Synthetic geochemical well logs generation using ensemble machine learning techniques for the Brazilian pre-salt reservoirs
        # de Oliveira, Lucas Abreu Blanes and de Carvalho Carneiro, Cleyton, [2021]
        svm = SVM(n_features,
                  device=None,
                  kernel='rbf',
                  gamma='scale',
                  C=1.0,
                  epsilon=0.1
                 )
    
        return svm
    
    @staticmethod
    def instantiate_MEAN(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        pass
    
    
    def __call__(self, seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir):
        return getattr(self, f"instantiate_{self.model.upper()}")(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir)
    
    
def instantiate(model: str, seq_len: int = 256, n_features: int = 4, 
                batch_size:int  = 32, epochs: int  = 50, patience: int = 15, 
                optimizer = None,
                device: str = 'cpu', output_dir: str = '.') -> Factory:
    '''
    Functional version of the Factory class. See above
    '''
    f = Factory(model)
    
    return f(seq_len, n_features, batch_size, epochs, patience, optimizer, device, output_dir)
