def get_atom_name_from_dir(job_dir):
    job_name = job_dir.split("/")[-2]
    name_shards = job_name.split("_")
    return "_".join(name_shards[4:-1])


def get_model_path_from_info_path(info_path):
    shards = info_path.split(".")[0].split("_")
    idx = shards.index("info")
    return '_'.join(shards[0:idx]) + '/model_' + '_'.join(shards[idx+1:]) + ".pkl"
