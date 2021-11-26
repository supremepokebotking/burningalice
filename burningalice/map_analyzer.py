mapping_config = {
    'rows': 10,
    'cols': 10,
    'empty_cell_color': '#ff00ff',
    'color_mapping':
    {
        'active_monster': '#ff00ff',
        'enemy_active_monster': '#ff00ff',
        'active_stats_modifier_monster': '#ff00ff',
        'enemy_active_stats_modifier_monster': '#ff00ff',
        'p1_field': '#ff00ff',
        'p2_field': '#ff00ff',
    }
}

cell_data = [
    {
        'row': 1,
        'col': 1,
        'type': 'active_monster',
        'name': 'Pikachu',
        'details': '"stats":{"atk":254,"def":206,"spa":134,"spd":174,"spe":150}',
        'actions': ['Thunder', 'Thunder Wave', 'Thundershock', 'Agility', ],
    }
]

class MapAnalyzer():
    def __init__(self, mapping_config):

        self.mapping_config = mapping_config
        self.cell_data = {}


    def process_frame(self, frame_key, cell_data):

        # resetooo
        self.cell_data[frame_key] = cell_data
