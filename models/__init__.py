from gym.envs.registration import register


register(
    id = 'WindGridWorld-v0',
    entry_point = 'models.wind_grid_world:WindGridWorld',
)

register(
    id = 'CliffGridWorld-v0',
    entry_point = 'models.cliff_grid_world:CliffGridWorld',
)
