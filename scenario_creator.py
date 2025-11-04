"""
Scenario creation: sites, cells, UEs initialization
Matches MATLAB scenario configurations exactly
"""

import numpy as np
import math
import json
from typing import List, Dict, Any
from fiveg_objects import Site, Cell, UE, SimulationParams


def load_scenario_config(scenario_name: str) -> SimulationParams:
    """Load scenario configuration from JSON file or predefined"""
    try:
        with open(f'scenarios/{scenario_name}.json', 'r') as f:
            config = json.load(f)
        return _parse_scenario_config(config)
    except FileNotFoundError:
        return _get_predefined_scenario(scenario_name)


def _parse_scenario_config(config: Dict[str, Any]) -> SimulationParams:
    """Parse JSON config into SimulationParams"""
    return SimulationParams(**config)


def _get_predefined_scenario(scenario_name: str) -> SimulationParams:
    """Get predefined scenario parameters"""
    scenarios = {
        'indoor_hotspot': _indoor_hotspot_params(),
        'dense_urban': _dense_urban_params(),
        'rural': _rural_params(),
        'urban_macro': _urban_macro_params(),
        'high_speed': _high_speed_params(),
        'extreme_rural': _extreme_rural_params(),
        'highway': _highway_params(),
    }
    
    if scenario_name in scenarios:
        return scenarios[scenario_name]
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")


def _indoor_hotspot_params() -> SimulationParams:
    """Indoor hotspot scenario parameters"""
    return SimulationParams(
        name="indoor_hotspot", deploymentScenario="indoor_hotspot",
        numSites=12, numSectors=1, isd=30, antennaHeight=3, cellRadius=50,
        carrierFrequency=3.5e9, numUEs=80, ueSpeed=3, minTxPower=20, maxTxPower=30,
        basePower=400, idlePower=100, max_radius=100
    )

def _dense_urban_params() -> SimulationParams:
    """Dense urban scenario parameters"""
    return SimulationParams(
        name="dense_urban", deploymentScenario="dense_urban",
        numSites=7, numSectors=3, isd=200, antennaHeight=25, cellRadius=200,
        numUEs=210, ueSpeed=3, indoorRatio=0.8, outdoorSpeed=30, max_radius=500
    )

def _rural_params() -> SimulationParams:
    """Rural scenario parameters"""
    return SimulationParams(
        name="rural", deploymentScenario="rural",
        numSites=7, numSectors=3, isd=500, antennaHeight=35, cellRadius=1000,
        minTxPower=35, maxTxPower=49, basePower=1200, idlePower=300,
        numUEs=100, ueSpeed=60, max_radius=2000
    )

def _urban_macro_params() -> SimulationParams:
    """Urban macro scenario parameters"""
    return SimulationParams(
        name="urban_macro", deploymentScenario="urban_macro",
        numSites=7, numSectors=3, isd=300, antennaHeight=25, cellRadius=300,
        numUEs=250, ueSpeed=30, indoorRatio=0.8, max_radius=800
    )

def _high_speed_params() -> SimulationParams:
    """High-speed railway scenario parameters"""
    return SimulationParams(
        name="high_speed", deploymentScenario="high_speed",
        numSites=10, numSectors=2, isd=1000, antennaHeight=35, cellRadius=1000,
        minTxPower=40, maxTxPower=49, basePower=1200, idlePower=300,
        numUEs=300, ueSpeed=500, trainLength=200, trackLength=10000, max_radius=1500
    )

def _extreme_rural_params() -> SimulationParams:
    """Extreme rural scenario parameters"""
    return SimulationParams(
        name="extreme_rural", deploymentScenario="extreme_rural",
        numSites=3, numSectors=3, isd=5000, antennaHeight=45, cellRadius=50000,
        minTxPower=43, maxTxPower=49, basePower=1500, idlePower=400,
        numUEs=50, ueSpeed=120, max_radius=50000
    )

def _highway_params() -> SimulationParams:
    """Highway scenario parameters"""
    return SimulationParams(
        name="highway", deploymentScenario="highway",
        numSites=12, numSectors=3, isd=866, antennaHeight=35, cellRadius=866,
        minTxPower=40, maxTxPower=49, basePower=1200, idlePower=300,
        numUEs=200, ueSpeed=120, highway_length=10000, num_lanes=3, lane_width=3.5, max_radius=1732
    )


def create_sites(params: SimulationParams, seed: int) -> List[Site]:
    """Create site layout based on deployment scenario"""
    np.random.seed(seed + 1000)
    scenario = params.deploymentScenario
    
    if scenario == 'indoor_hotspot':
        return _create_indoor_layout(params)
    elif scenario in ['dense_urban', 'rural', 'urban_macro', 'extreme_rural']:
        site_type_map = {
            'dense_urban': 'macro', 'rural': 'rural_macro',
            'urban_macro': 'urban_macro', 'extreme_rural': 'extreme_rural_macro'
        }
        return _create_hex_layout(params, site_type_map[scenario])
    elif scenario == 'high_speed':
        return _create_high_speed_layout(params)
    elif scenario == 'highway':
        return _create_highway_layout(params)
    else:
        return _create_hex_layout(params, 'macro')

def _create_indoor_layout(params: SimulationParams) -> List[Site]:
    """Create indoor grid layout"""
    sites = []
    site_id = 1
    for row in range(1, 4):
        for col in range(1, 5):
            if site_id <= params.numSites:
                sites.append(Site(id=site_id, x=col * 24, y=row * 12.5, type='indoor_trxp'))
                site_id += 1
    return sites

def _create_hex_layout(params: SimulationParams, site_type: str) -> List[Site]:
    """Create hexagonal layout"""
    sites = [Site(id=1, x=0, y=0, type=site_type)]
    if params.numSites == 1: return sites
    
    site_idx = 2
    ring = 1
    while site_idx <= params.numSites:
        ring_sites = _create_hex_ring(ring, params.isd, site_idx, site_type)
        for site in ring_sites:
            if site_idx <= params.numSites:
                sites.append(site)
                site_idx += 1
        ring += 1
    return sites

def _create_hex_ring(ring: int, isd: float, start_idx: int, site_type: str) -> List[Site]:
    """Create one ring of hexagonal layout"""
    ring_sites = []
    site_idx = start_idx
    for side in range(6):
        for pos in range(ring):
            angle = side * math.pi / 3
            x = isd * ring * math.cos(angle) + pos * isd * math.cos(angle + math.pi / 3)
            y = isd * ring * math.sin(angle) + pos * isd * math.sin(angle + math.pi / 3)
            ring_sites.append(Site(id=site_idx, x=x, y=y, type=site_type))
            site_idx += 1
    return ring_sites

def _create_high_speed_layout(params: SimulationParams) -> List[Site]:
    """Create linear layout along railway track"""
    start_pos = -(params.numSites - 1) * params.isd / 2
    return [Site(id=i+1, x=start_pos + i * params.isd, y=100, type='high_speed_rrh') for i in range(params.numSites)]

def _create_highway_layout(params: SimulationParams) -> List[Site]:
    """Create linear layout along highway"""
    spacing = params.highway_length / (params.numSites - 1)
    start_pos = -params.highway_length / 2
    return [Site(id=i+1, x=start_pos + i * spacing, y=50 * ((i % 2) * 2 - 1), type='highway_macro') for i in range(params.numSites)]


def configure_cells(sites: List[Site], params: SimulationParams) -> List[Cell]:
    """Configure cells for all sites"""
    cells = []
    cell_id = 1
    
    site_type_map = {
        'indoor_trxp': (_get_indoor_cell_config, 1), 'high_speed_rrh': (_get_high_speed_cell_config, 2),
        'rural_macro': (_get_rural_macro_cell_config, params.numSectors), 'urban_macro': (_get_urban_macro_cell_config, params.numSectors),
        'extreme_rural_macro': (_get_extreme_rural_cell_config, params.numSectors), 'highway_macro': (_get_highway_cell_config, params.numSectors),
        'macro': (_get_macro_cell_config, params.numSectors)
    }

    for site in sites:
        config_func, num_sectors = site_type_map.get(site.type, (_get_macro_cell_config, params.numSectors))
        cell_config = config_func(params)
        
        for sector_id in range(1, num_sectors + 1):
            cells.append(Cell(
                id=cell_id, site_id=site.id, sector_id=sector_id,
                azimuth=(sector_id - 1) * (360 / num_sectors), x=site.x, y=site.y,
                is_omnidirectional=(num_sectors == 1), site_type=site.type, **cell_config
            ))
            cell_id += 1
    return cells

def _get_common_cell_config(params: SimulationParams) -> Dict[str, Any]:
    return {'frequency': params.carrierFrequency, 'min_tx_power': params.minTxPower, 'max_tx_power': params.maxTxPower}

def _get_indoor_cell_config(params: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_cell_config(params), 'antenna_height': 3, 'tx_power': 23, 'base_energy_consumption': 400, 'idle_energy_consumption': 100, 'max_capacity': 50, 'ttt': 4, 'a3_offset': 6}
def _get_macro_cell_config(params: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_cell_config(params), 'antenna_height': 25, 'tx_power': 43, 'base_energy_consumption': 800, 'idle_energy_consumption': 200, 'max_capacity': 200, 'ttt': 8, 'a3_offset': 8}
def _get_rural_macro_cell_config(params: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_cell_config(params), 'antenna_height': 35, 'tx_power': 46, 'base_energy_consumption': 1200, 'idle_energy_consumption': 300, 'max_capacity': 150, 'ttt': 12, 'a3_offset': 10}
def _get_urban_macro_cell_config(params: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_cell_config(params), 'antenna_height': 25, 'tx_power': 43, 'base_energy_consumption': 1000, 'idle_energy_consumption': 250, 'max_capacity': 250, 'ttt': 8, 'a3_offset': 8}
def _get_high_speed_cell_config(params: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_cell_config(params), 'antenna_height': 35, 'tx_power': 46, 'base_energy_consumption': 1200, 'idle_energy_consumption': 300, 'max_capacity': 300, 'ttt': 0.04, 'a3_offset': 3}
def _get_extreme_rural_cell_config(params: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_cell_config(params), 'antenna_height': 45, 'tx_power': 46, 'base_energy_consumption': 1500, 'idle_energy_consumption': 400, 'max_capacity': 100, 'ttt': 16, 'a3_offset': 12}
def _get_highway_cell_config(params: SimulationParams) -> Dict[str, Any]:
    return {**_get_common_cell_config(params), 'antenna_height': 35, 'tx_power': 46, 'base_energy_consumption': 1200, 'idle_energy_consumption': 300, 'max_capacity': 300, 'ttt': 0.04, 'a3_offset': 3}


def initialize_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize UEs based on deployment scenario"""
    np.random.seed(seed + 2000)
    init_func = {
        'indoor_hotspot': _initialize_indoor_ues, 'dense_urban': _initialize_dense_urban_ues,
        'rural': _initialize_rural_ues, 'urban_macro': _initialize_urban_macro_ues,
        'high_speed': _initialize_high_speed_ues, 'extreme_rural': _initialize_extreme_rural_ues,
        'highway': _initialize_highway_ues
    }.get(params.deploymentScenario, _initialize_default_ues)
    return init_func(params, sites, seed)

def _initialize_indoor_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize indoor UEs"""
    mobility = {'stationary': (0, 0.4), 'slow_walk': (0.5, 0.4), 'normal_walk': (1.5, 0.2)}
    patterns = list(mobility.keys())
    props = list(mobility.values())
    velocities, weights = zip(*props)
    
    ues = []
    for ue_id in range(1, params.numUEs + 1):
        idx = np.random.choice(len(patterns), p=weights)
        ues.append(UE(
            id=ue_id, x=10 + np.random.random() * 100, y=5 + np.random.random() * 40,
            velocity=velocities[idx], direction=np.random.random() * 2 * math.pi,
            mobility_pattern=patterns[idx], rng_seed=seed + ue_id * 100, deployment_scenario=params.deploymentScenario
        ))
    return ues

def _initialize_dense_urban_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize dense urban UEs (mix of indoor/outdoor)"""
    ues = []
    for ue_id in range(1, params.numUEs + 1):
        site = sites[np.random.randint(len(sites))]
        angle = np.random.random() * 2 * math.pi
        is_indoor = ue_id <= int(params.numUEs * params.indoorRatio)
        
        distance = abs(np.random.randn()) * 30 if is_indoor else 50 + np.random.random() * 100
        velocity = params.ueSpeed / 3.6 if is_indoor else params.outdoorSpeed / 3.6
        pattern = 'indoor_pedestrian' if is_indoor else 'outdoor_vehicle'
            
        ues.append(UE(
            id=ue_id, x=site.x + distance * math.cos(angle), y=site.y + distance * math.sin(angle),
            velocity=velocity, direction=np.random.random() * 2 * math.pi, mobility_pattern=pattern,
            rng_seed=seed + ue_id * 100, deployment_scenario=params.deploymentScenario
        ))
    return ues

def _initialize_rural_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize rural UEs"""
    mobility = {'stationary': (0, 0.1), 'pedestrian': (1.0, 0.4), 'slow_vehicle': (30/3.6, 0.3), 'fast_vehicle': (60/3.6, 0.2)}
    patterns, props = zip(*mobility.items())
    velocities, weights = zip(*props)
    
    ues = []
    for ue_id in range(1, params.numUEs + 1):
        if np.random.random() < 0.6:
            site = sites[np.random.randint(len(sites))]
            angle, distance = np.random.random() * 2 * math.pi, np.random.random() * 200
            x, y = site.x + distance * math.cos(angle), site.y + distance * math.sin(angle)
        else:
            angle, radius = np.random.random() * 2 * math.pi, params.max_radius * math.sqrt(np.random.random())
            x, y = radius * math.cos(angle), radius * math.sin(angle)
        
        idx = np.random.choice(len(patterns), p=weights)
        ues.append(UE(
            id=ue_id, x=x, y=y, velocity=velocities[idx], direction=np.random.random() * 2 * math.pi,
            mobility_pattern=patterns[idx], rng_seed=seed + ue_id * 100, deployment_scenario=params.deploymentScenario
        ))
    return ues

def _initialize_urban_macro_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize urban macro UEs"""
    mobility = {'pedestrian': (1.5, 0.6), 'slow_vehicle': (15/3.6, 0.2), 'vehicle': (30/3.6, 0.2)}
    patterns, props = zip(*mobility.items())
    velocities, weights = zip(*props)
    
    ues = []
    for ue_id in range(1, params.numUEs + 1):
        site = sites[np.random.randint(len(sites))]
        angle = np.random.random() * 2 * math.pi
        distance = abs(np.random.randn()) * (params.cellRadius * 0.3) if np.random.random() < params.indoorRatio else params.cellRadius * math.sqrt(np.random.random())
        idx = np.random.choice(len(patterns), p=weights)
        
        ues.append(UE(
            id=ue_id, x=site.x + distance * math.cos(angle), y=site.y + distance * math.sin(angle),
            velocity=velocities[idx], direction=np.random.random() * 2 * math.pi, mobility_pattern=patterns[idx],
            rng_seed=seed + ue_id * 100, deployment_scenario=params.deploymentScenario
        ))
    return ues

def _initialize_high_speed_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize high-speed train UEs"""
    start_x = -params.trackLength / 2
    ues = []
    for ue_id in range(1, params.numUEs + 1):
        pos_in_train = (ue_id - 1) / max(1, params.numUEs - 1) * params.trainLength
        ues.append(UE(
            id=ue_id, x=start_x + pos_in_train, y=(np.random.random() - 0.5) * 4,
            velocity=params.ueSpeed / 3.6, direction=0, mobility_pattern='high_speed_train',
            rng_seed=seed + ue_id * 100, deployment_scenario=params.deploymentScenario, in_train=True,
            track_length=params.trackLength, train_start_x=start_x, position_in_train=pos_in_train
        ))
    return ues

def _initialize_extreme_rural_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize extreme rural UEs"""
    mobility = {'stationary': (0, 0.1), 'slow_vehicle': (60/3.6, 0.4), 'fast_vehicle': (120/3.6, 0.4), 'extreme_vehicle': (160/3.6, 0.1)}
    patterns, props = zip(*mobility.items())
    velocities, weights = zip(*props)
    
    ues = []
    for ue_id in range(1, params.numUEs + 1):
        angle, radius = np.random.random() * 2 * math.pi, params.max_radius * math.sqrt(np.random.random())
        idx = np.random.choice(len(patterns), p=weights)
        ues.append(UE(
            id=ue_id, x=radius * math.cos(angle), y=radius * math.sin(angle), velocity=velocities[idx],
            direction=np.random.random() * 2 * math.pi, mobility_pattern=patterns[idx],
            rng_seed=seed + ue_id * 100, deployment_scenario=params.deploymentScenario
        ))
    return ues

def _initialize_highway_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Initialize highway UEs"""
    ues = []
    for ue_id in range(1, params.numUEs + 1):
        is_forward = np.random.random() < 0.5
        lane = np.random.randint(1, params.num_lanes + 1)
        y = ((lane - 0.5) * params.lane_width) * (1 if is_forward else -1)
        
        ues.append(UE(
            id=ue_id, x=np.random.random() * params.highway_length - params.highway_length/2, y=y,
            velocity=params.ueSpeed/3.6 + (np.random.random() - 0.5) * 20,
            direction=0 if is_forward else math.pi, mobility_pattern='highway_vehicle',
            rng_seed=seed + ue_id * 100, deployment_scenario=params.deploymentScenario, on_highway=True,
            highway_length=params.highway_length, num_lanes=params.num_lanes,
            lane_width=params.lane_width, lane=lane, is_forward=is_forward
        ))
    return ues

def _initialize_default_ues(params: SimulationParams, sites: List[Site], seed: int) -> List[UE]:
    """Default UE initialization (similar to dense urban)"""
    return _initialize_dense_urban_ues(params, sites, seed)