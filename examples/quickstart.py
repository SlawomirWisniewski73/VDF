from vdf import VDFStream, Header

def main():
    # Stan początkowy (S0)
    S0 = [0, 0, 0, 0]
    vdf = VDFStream(S0, header=Header(dimensions=[4], checkpoint_interval=2))

    # Dodajemy kilka deltas
    vdf.append_delta([1], [5], op="set")   # Δ1: ustaw element 1 na 5
    vdf.append_delta([2], [3], op="add")   # Δ2: dodaj 3 do elementu 2 (checkpoint)
    vdf.append_delta([3], [2], op="mul")   # Δ3: pomnóż element 3 przez 2

    # Rekonstrukcja stanów S0..S3
    for k in range(4):
        state = vdf.get_state(k)
        print(f"S_{k} = {state}")

if __name__ == "__main__":
    main()
