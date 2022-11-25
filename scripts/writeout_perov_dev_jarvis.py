import pandas as pd

from pymatgen.core.structure import Structure

def main():
    for phase in ["train","val","test"]:
        print("reading csv...")
        perov = pd.read_csv(f"~/Projects/mila/cdvae/data/perov_5/{phase}.csv", low_memory=False, index_col="Unnamed: 0")

        crystal_from_str = lambda crystal_str: Structure.from_str(crystal_str, fmt='cif')
        crystal_to_coords = lambda crystal: str(crystal.cart_coords.tolist())
        crystal_to_atoms = lambda crystal: str(list(crystal.atomic_numbers))
        wrap_trans = lambda crystal: {
            "coords": crystal_to_coords(crystal),
            "elements": crystal_to_atoms(crystal)
        }

        wrap_full = lambda crystal_str:  wrap_trans(crystal_from_str(crystal_str))

        results = perov["cif"].apply(wrap_full)
        results = pd.DataFrame(list(results))
        perov["coords"] = results.coords.values
        perov["elements"] = results.elements.values

        del perov["cif"]

        perov["num_atoms"] = perov["elements"].apply(len)

        perov = perov.rename(columns={"material_id":"dataset_id"})
        print("writing to disk...")
        perov.to_csv(f"~/Projects/mila/cdvae/data/perov_j/{phase}.csv", index=False)


if __name__ == "__main__":
    main()
