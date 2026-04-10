from collections import namedtuple

def attribute(device):

    dataset_name = 'LGES_sample'
    n_epochs = 500
    latent_size = 128
    pre_accumulation_steps = 10

    message_passing_steps = 10

    monitor_interval = 25000
    test_interval = 25000
    test_length = 293
    test_sequence_idx = 5
    
    # 원격 GPU server
    ds_path = "/home/ssdl/PJW/Particle_Simulation/training_40_/"
    training_path = "train/"
    testing_path = "test/"
    calculix_path = "/home/ssdl/PJW/Particle_Simulation/calculix/"
    
    # 연구실
    """
    ds_path = r"C:/Users/AISDL_PJW/Particle_Simulation/training_40_/"
    training_path = "train/"
    testing_path = "test/"
    calculix_path = r"C:/Users/AISDL_PJW/Particle_Simulation/calculix/"
    """

    training_noise = 0.0003
    contact_distance = 0.75

    lr = 1e-4
    decay_offset = 1e-1 # expointential decay from lr to lr*decay_offset
    secondary_decay_offset = 1e-2 # linear decay from lr*decay_offset to lr*secondary_decay_offset
    lr_decay_length = 8e5
    norm_acc_length = 1e6
    grad_limit = 3.0 # normalizing gradients if limit is exceeded

    num_history = 5

    TrainingParameterPack = namedtuple('trainingparameterpack', ['dataset_name','nepochs','latent_size','pre_accumulation_steps','message_passing_steps'])
    trainingparameterpack = TrainingParameterPack(dataset_name, n_epochs, latent_size, pre_accumulation_steps, message_passing_steps)

    TestingParameterPack = namedtuple('testingparameterpack', ['monitor_interval','test_interval','test_length','test_sequence_idx'])
    testingparameterpack = TestingParameterPack(monitor_interval,test_interval,test_length,test_sequence_idx)

    DataParameterPack = namedtuple('dataparameterpack', ['ds_path','training_path','testing_path', 'calculix_path', 'training_noise','contact_distance', 'num_history'])
    dataparameterpack = DataParameterPack(ds_path,training_path,testing_path,calculix_path,training_noise,contact_distance, num_history)

    OptimizerParameterPack = namedtuple('optimizerparameterpack', ['lr','decay_offset','secondary_decay_offset','lr_decay_length','norm_acc_length','grad_limit'])
    optimizerparameterpack = OptimizerParameterPack(lr, decay_offset, secondary_decay_offset, lr_decay_length, norm_acc_length, grad_limit)

    trainingattributespack = [trainingparameterpack,
                                testingparameterpack,
                                dataparameterpack,
                                optimizerparameterpack]
    
    NetworkParamterPack = namedtuple('networkparameterpack', ['name', 'n_layers', 'input_size', 'hidden_size', 'output_size', 'norm', 'multi_mlp_cnt', 'bias', 'residual', 'device'])

    edge_encoder_pack = NetworkParamterPack('edge_encoder', 3, 7, latent_size, latent_size, True, 1, False, False, device)
    node_encoder_pack = NetworkParamterPack('node_encoder', 3, 25, latent_size, latent_size, True, 1, False, False, device)

    edge_messenger_pack = NetworkParamterPack('edge_messenger', 3, latent_size * 3, latent_size, latent_size, True, message_passing_steps, False, True, device)
    edge_attention_pack = NetworkParamterPack('edge_attention', 3, latent_size*3, latent_size, 1, False, 1, False, False, device)
    node_messenger_pack = NetworkParamterPack('node_messenger', 3, latent_size * 2, latent_size, latent_size, True, message_passing_steps, False, True, device)
    #decoder_pack = NetworkParamterPack('decoder', 3, latent_size, latent_size, 3, False, 1, True, False, device)
    edge_decoder_pp_pack = NetworkParamterPack('edge_decoder_pp', 3, latent_size, latent_size, 3, False, 1, True, False, device)
    edge_decoder_pm_pack = NetworkParamterPack('edge_decoder_pm', 3, latent_size, latent_size, 3, False, 1, True, False, device)

    NetworkAttributesPack = namedtuple('networkattributespack', ['edge_encoder',
                                                                 'node_encoder',
                                                                 'edge_messenger',
                                                                 'edge_attention',
                                                                 'node_messenger',
                                                                 'edge_decoder_pp',   # 변경
                                                                 'edge_decoder_pm'])  # 변경
    
    networkattributespack = NetworkAttributesPack(edge_encoder_pack,
                                                  node_encoder_pack,
                                                  edge_messenger_pack,
                                                  edge_attention_pack,
                                                  node_messenger_pack,
                                                  edge_decoder_pp_pack,   # 변경
                                                  edge_decoder_pm_pack)   # 변경

    return networkattributespack, trainingattributespack

