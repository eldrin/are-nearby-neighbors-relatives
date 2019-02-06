def within_space_consistency(filtered_data, sample=None, verbose=False):
    """"""
    d = filtered_data.sample(frac=sample)  # aliasing
    d = d[d['to_perturbation'] == 'original']
    
    output = {}
    targets = d['from'].unique()
    if verbose:
        targets = tqdm(targets, ncols=80)
        
    for target in targets:
        try:
            # retrieve target
            d_t = d[d['from'] == target]

            # get comparisons
            d_ts_s = d_t[d_t['to'] == target]['distance'].values
            if len(d_ts_s) == 0:
                raise ValueError()
            d_ts_s_prime = d_t[d_t['to'] != target]['distance'].values

            # save error
            output[target] = not np.all(d_ts_s < d_ts_s_prime)

        except Exception as e:
            output[target] = np.NaN
        
    return output