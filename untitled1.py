def loadTrainTest_custom_randz(Testset, sim = 'des', train_dirIn = '/data/a/cpac/nramachandra/Projects/phoZ/Synthetic_Data/fsps_wrapper/notebooks/out/rand_z/', nbands = 5, frac_train = 0.9):

    print("Training on sim: ", sim)
    
    # Load in (raw) training data, reshape
    '''
    col_mag = np.load(train_dirIn + 'col_mag_' + sim + '_0.npy')
    nprop, ngal, nz, ncol = col_mag.shape
    fsps_reshaped = col_mag.reshape(-1, nz, ncol) # Keep the last two dimensions, crush the others
    fsps_reshaped = fsps_reshaped.reshape(-1, ncol) # And then crush the remaining one?
    
    zz_all = np.load(train_dirIn + 'zz_'+ sim +'.npy')
    ngal, nprop, nz = zz_all.shape
    zz_all_reshaped = zz_all.reshape(-1, nz)
    zz_all_reshaped = zz_all_reshaped.reshape(-1)
    '''
    if sim == "irac" or sim == "wise":
        D = 4
    elif sim == "des" or sim == "sdss":
        D = 5
    elif sim == "lsst":
        D = 6
    elif sim == "jwst":
        D = 8
    elif sim == "pau":
        D = 46 # Should this be 40?
    elif sim == "spherex":
        D = 102
        
    fsps_reshaped = np.zeros(shape=(1, nbands))
    zz_reshaped = np.zeros(shape=(1))

    for seed_id in range(100):

        col_mag = np.load(train_dirIn + 'col_mag_z_' + str(seed_id)+ sim + '_0.npy')
        # ngal, nprop, nz, ncol = col_mag.shape
        nprop, ngal, nz, ncol = col_mag.shape

        fsps_reshaped0 = col_mag.reshape(-1, nz, ncol)
        fsps_reshaped = np.append( fsps_reshaped, fsps_reshaped0.reshape(-1, ncol), axis=0)

        zz_all = np.load(train_dirIn + 'zz_z_'+ str(seed_id) + sim +'.npy')
        #print(zz_all.shape)

        ngal, nprop, nz = zz_all.shape
        zz_reshaped0 = zz_all.reshape(-1, nz)
        zz_reshaped = np.append( zz_reshaped, zz_reshaped0.reshape(-1) )

    fsps_reshaped = fsps_reshaped[1:, :]
    zz_all_reshaped = zz_reshaped[1:]

    
    print("X training data shape: ", fsps_reshaped.shape)
    print("y training data shape: ", zz_all_reshaped.shape)
    
    # Randomly sample 90% and 10% for training and testing
    sample_size = fsps_reshaped.shape[0] # first dimension
    train_n = int(frac_train*sample_size) # note: rounds down
    np.random.seed(0)
    full_idx = np.arange(sample_size) # indices for all datapoints, ordered
    training_idx = np.random.choice(full_idx, train_n, replace = False) # pick a subset of those indices for training
    X_train = fsps_reshaped[training_idx]
    y_train = zz_all_reshaped[training_idx]
    X_test = fsps_reshaped[full_idx[~np.isin(full_idx, training_idx)]]  # the indices not part of the subset will be used for testing
    y_test = zz_all_reshaped[full_idx[~np.isin(full_idx, training_idx)]]
    
    #X_err = np.load(test_dirIn + 'test_' + Testset +'_err.npy')
    #test_labels = np.load(test_dirIn + 'test_' + Testset + '_label.npy')
    
    print_limits(fsps_reshaped, zz_all_reshaped)
    print_limits(X_test, y_test)

    return X_train, y_train, X_test, y_test#, X_err, test_labels