def get_atom_name_from_dir(job_dir):
    job_name = job_dir.split("/")[-2]
    name_shards = job_name.split("_")
    return "_".join(name_shards[4:-1])
