
class JetGraphProducer(InMemoryDataset):

    def __init__(
        self,
        root,
        delta_r_threshold,
        n_store_jets,
        n_store_cands=None,
        max_events_to_process=None,
        use_delta_r_star=False,
        use_delta_r_star_star=False,
        transform=None,
        pre_transform=NormalizeFeatures(["x"]),
        pre_filter=None,
    ):

        self.root = root
        self.n_store_cands = n_store_cands
        self.n_store_jets = n_store_jets
        self.delta_r_threshold = delta_r_threshold
        self.max_events_to_process = max_events_to_process
        self.use_delta_r_star = use_delta_r_star
        self.use_delta_r_star_star = use_delta_r_star_star
        self.pre_transform = pre_transform

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.root + "/" + f for f in os.listdir(self.root) if f.endswith(".root")]

    @property
    def processed_file_names(self):
        return [f"processed_deltaR_{str(self.delta_r_threshold).replace('.', 'p')}.pt"]

    def get_graphs(self, file_name):
        with uproot.open(f"{file_name}:Events") as in_file:
            events = in_file.arrays([
                "PFCands_pt",
                "PFCands_eta",
                "PFCands_phi",
                "PFCands_pdgId",
                "PFCands_mass",
                "FatJetPFCands_jetIdx",
                "FatJetPFCands_pFCandsIdx",
                "nFatJet",
                "genWeight",
            ])

        graphDataset = []
        events_to_process = self.max_events_to_process if self.max_events_to_process else len(events)

        for i_ev in tqdm(range(events_to_process)):

            n_jets = min(self.n_store_jets, events.nFatJet[i_ev])

            for nj in range(n_jets):

                event = events[i_ev]
                genWeight = event.genWeight
                pf_cands_matching_filter = event["FatJetPFCands_pFCandsIdx"][event["FatJetPFCands_jetIdx"] == nj]
                pt = event["PFCands_pt"][pf_cands_matching_filter]
                eta = event["PFCands_eta"][pf_cands_matching_filter]
                phi = event["PFCands_phi"][pf_cands_matching_filter]
                pdgId = event["PFCands_pdgId"][pf_cands_matching_filter]
                mass = event["PFCands_mass"][pf_cands_matching_filter]

                # Order everything by pt and keep the desired number of candidates
                permutation = ak.argsort(pt, ascending=False)
                n_constituents = min(len(permutation), self.n_store_cands) if self.n_store_cands else len(permutation)
                pt = np.array(pt[permutation][:n_constituents])
                eta = np.array(eta[permutation][:n_constituents])
                phi = np.array(phi[permutation][:n_constituents])
                pdgId = np.array(pdgId[permutation][:n_constituents])
                mass = np.array(mass[permutation][:n_constituents])
                pos = [[e, p] for e, p in zip(eta, phi)]

                # Converting to np.array and subsequently to torch.tensor as suggested in torch docs for performance
                features = torch.tensor(np.array([
                    pt,
                    eta,
                    phi,
                    pdgId,
                    mass,
                ]).T)

                # Calculate edges
                matrix_eta = np.repeat(eta, len(eta)).reshape((len(eta), -1))
                matrix_phi = np.repeat(phi, len(phi)).reshape((len(phi), -1))
                delta_eta = matrix_eta - matrix_eta.T
                # Calculate delta phi accounting for circularity
                delta_phi_internal = np.abs(matrix_phi - matrix_phi.T)
                delta_phi_external = 2*np.pi - np.abs(matrix_phi - matrix_phi.T)
                delta_phi = np.minimum(delta_phi_internal, delta_phi_external)
                adjacency = (np.sqrt(delta_eta**2 + delta_phi**2) < self.delta_r_threshold).astype(int)
                edge_connections = np.where( (adjacency - np.identity(adjacency.shape[0])) == 1)
                edge_index = torch.tensor([ edge for edge in zip(edge_connections[0], edge_connections[1]) ], dtype=torch.long)

                # Build graph
                graphDataset.append(
                    Data(
                        x=features,
                        edge_index=edge_index.t().contiguous(),
                        num_nodes=n_constituents,
                        num_node_features=features.shape[1],
                        pos=pos,
                        y=torch.tensor([nj, genWeight]) # Holds the jet index and event weight for future use
                    )
                )

        if self.pre_transform is not None:
            graphDataset = [self.pre_transform(d) for d in graphDataset]

        return graphDataset


    def process(self):
        # Read data into huge `Data` list.
        graphs = []
        for file in self.raw_file_names:
            print(f"Processing {file}")
            graphs += self.get_graphs(file)
        
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[-1])
