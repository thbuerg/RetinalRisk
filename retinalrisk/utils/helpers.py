def extract_metadata(datamodule):
    
    records = datamodule.record_cols
    covariates = get_covariate_names(datamodule)
    endpoints = sorted(list(datamodule.label_mapping.values()))

    return records, covariates, endpoints

def get_covariate_names(datamodule):
    feature_names = []
    if len(datamodule.covariate_cols) == 0:
        return feature_names

    for name, trans, column, _ in datamodule.covariate_preprocessor._iter(fitted=True):
        try:
            feature_names += list(trans.get_feature_names_out())
        except AttributeError:
            # should be numerical, i.e. feature_name_in == feature_name_out
            feature_names += list(trans.feature_names_in_)
    return feature_names
