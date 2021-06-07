def resample(X, y, n_bins = 200):
    select_per_bin = 50 #np.int(num_train/n_bins) #100

    bins = np.linspace(y.min(), y.max(), n_bins)
    inds = np.digitize(y, bins)

    resampled_ind = []

    for ind_i in range(n_bins):
        ind_bin = np.where(inds==ind_i)
        random_choices = np.min( [select_per_bin, np.size(ind_bin) ])
        index = np.random.choice(ind_bin[0], random_choices, replace=False)
        resampled_ind = np.append(resampled_ind, index)

    resampled_ind = resampled_ind.astype('int')
    all_ind = np.arange(y.shape[0])

    plt.figure(23)
    plt.hist(y, density=True, bins = n_bins, histtype='step', label='original')
    y_resampled = y[resampled_ind]
    X_resampled = X[resampled_ind]

    plt.hist(y_resampled, density=True, bins = n_bins, histtype='step', label='resampled')
    plt.hist(y_train, density=True, bins = n_bins, histtype='step', label='rest')

    plt.legend()
    plt.show()

    print(y_resampled.shape)
    return X_resampled, y_resampled, resampled_ind


def loadTrainTest_july(dirIn = 'Data/fromGalaxev/photozs/datasets/data_july_2020/'):
    train_data = np.load(dirIn + 'july14_200k.npy') 
    test_data = np.load(dirIn + 'july13_10k.npy') 

    X_train = train_data['color']#[train_data['redshift_flags'] == 0]
    y_train = train_data['redshift']#[train_data['redshift_flags'] == 0]

    X_test = test_data['color']#[test_data['redshift_flags'] == 0]
    y_test = test_data['redshift']#[test_data['redshift_flags'] == 0]

    print_limits(X_train, y_train)
    print_limits(X_test, y_test)

    return X_train, y_train, X_test, y_test

X_train, y_train, _, _ = loadTrainTest_july(dirIn = 'Data/fromGalaxev/photozs/datasets/data_july_2020/')
X_train, y_train, resampled_ind = resample(X_train, y_train)

