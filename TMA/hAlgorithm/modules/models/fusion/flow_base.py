class FlowBase:
    def __init__(self, mf_frame_num) -> None:
        self.hidden = []
        self.mf_frame_num = mf_frame_num

    def clear_hidden(self):
        self.hidden = []

    def prepare_flow(self, feat_list):
        assert len(feat_list) == 1
        feat_flow = []
        self.hidden = self.hidden[-self.mf_frame_num + 1 :]
        self.hidden.extend(feat_list)
        feat_flow.extend(self.hidden)
        while len(feat_flow) < self.mf_frame_num:
            feat_flow.extend(feat_list)
        return feat_flow
