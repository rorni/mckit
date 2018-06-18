# input for elements
element_data = [
    ('H', {}), ('1000', {}), ('1001', {}), 
    ('CA', {}), ('ca', {}), ('Ca', {'lib': '21c'}), ('CA40', {'lib': '21c'}), ('CA-40', {}), ('Ca42', {'lib': '21c'}), 
    ('ca-43', {}), ('CA-41', {}), 
    ('U', {}), ('U', {'isomer': 1}), ('U235', {}), ('u235', {'isomer': 1, 'lib': '50c'}),
    ('U-238', {'comment': 'pure 238'}), ('92238', {}), ('92000', {}), 
    ('Be', {}), ('Be-9', {}), ('4000', {}), ('4009', {}), (4000, {}), (4009, {})
]

# Element creation cases. ans params.
creation_cases = [
    {'charge': 1, 'mass_number': 0, 'molar_mass': 1.0079, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 1, 'mass_number': 0, 'molar_mass': 1.0079, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 1, 'mass_number': 1, 'molar_mass': 1.007825, 'lib': None, 'isomer': 0, 'comment': None},
    
    {'charge': 20, 'mass_number': 0, 'molar_mass': 40.078, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 20, 'mass_number': 0, 'molar_mass': 40.078, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 20, 'mass_number': 0, 'molar_mass': 40.078, 'lib': '21c', 'isomer': 0, 'comment': None},
    {'charge': 20, 'mass_number': 40, 'molar_mass': 39.962591, 'lib': '21c', 'isomer': 0, 'comment': None},
    {'charge': 20, 'mass_number': 40, 'molar_mass': 39.962591, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 20, 'mass_number': 42, 'molar_mass': 41.958618, 'lib': '21c', 'isomer': 0, 'comment': None},
    {'charge': 20, 'mass_number': 43, 'molar_mass': 42.958767, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 20, 'mass_number': 41, 'molar_mass': 41, 'lib': None, 'isomer': 0, 'comment': None},
    
    {'charge': 92, 'mass_number': 0, 'molar_mass': 238.0289, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 92, 'mass_number': 0, 'molar_mass': 238.0289, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 92, 'mass_number': 235, 'molar_mass': 235.043923, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 92, 'mass_number': 235, 'molar_mass': 235.043923, 'lib': '50c', 'isomer': 1, 'comment': None},
    {'charge': 92, 'mass_number': 238, 'molar_mass': 238.050783, 'lib': None, 'isomer': 0, 'comment': 'pure 238'},
    {'charge': 92, 'mass_number': 238, 'molar_mass': 238.050783, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 92, 'mass_number': 0, 'molar_mass': 238.0289, 'lib': None, 'isomer': 0, 'comment': None},
    
    {'charge': 4, 'mass_number': 0, 'molar_mass': 9.012182, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 4, 'mass_number': 9, 'molar_mass': 9.012182, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 4, 'mass_number': 0, 'molar_mass': 9.012182, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 4, 'mass_number': 9, 'molar_mass': 9.012182, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 4, 'mass_number': 0, 'molar_mass': 9.012182, 'lib': None, 'isomer': 0, 'comment': None},
    {'charge': 4, 'mass_number': 9, 'molar_mass': 9.012182, 'lib': None, 'isomer': 0, 'comment': None}
]

# The result of expansion from creation_cases.
expand_cases = [
    [('H1', {}, 0.999885), ('H2', {}, 0.000115)],
    [('H1', {}, 0.999885), ('H2', {}, 0.000115)],
    [('H1', {}, 1.0)],
    
    [('CA40', {}, 0.96941), ('CA42', {}, 0.00647), ('CA43', {}, 0.00135), ('CA44', {}, 0.02086), ('CA46', {}, 0.00004), ('CA48', {}, 0.00187)],
    [('CA40', {}, 0.96941), ('CA42', {}, 0.00647), ('CA43', {}, 0.00135), ('CA44', {}, 0.02086), ('CA46', {}, 0.00004), ('CA48', {}, 0.00187)],
    [('CA40', {'lib': '21c'}, 0.96941), ('CA42', {'lib': '21c'}, 0.00647), ('CA43', {'lib': '21c'}, 0.00135), ('CA44', {'lib': '21c'}, 0.02086), ('CA46', {'lib': '21c'}, 0.00004), ('CA48', {'lib': '21c'}, 0.00187)],
    [('CA40', {'lib': '21c'}, 1.0)],
    [('CA40', {}, 1.0)],
    [('CA42', {'lib': '21c'}, 1.0)],
    [('CA43', {}, 1.0)],
    [],
    
    [('U234', {}, 0.000055), ('U235', {}, 0.007200),  ('U238', {}, 0.992745)],
    [('U234', {}, 0.000055), ('U235', {}, 0.007200),  ('U238', {}, 0.992745)],
    [('U235', {}, 1.0)],
    [('U235', {'isomer': 1, 'lib': '50c'}, 1.0)],
    [('U238', {}, 1.0)],
    [('U238', {}, 1.0)],
    [('U234', {}, 0.000055), ('U235', {}, 0.007200),  ('U238', {}, 0.992745)],
    
    [('BE9', {}, 1.0)],
    [('BE9', {}, 1.0)],
    [('BE9', {}, 1.0)],
    [('BE9', {}, 1.0)],
    [('BE9', {}, 1.0)],
    [('BE9', {}, 1.0)]
]

# String representation for creation cases
str_cases = [
    'H', 'H', 'H1', 
    'Ca', 'Ca', 'Ca', 'Ca40', 'Ca40', 'Ca42', 'Ca43', 'Ca41',
    'U', 'U', 'U235', 'U235m', 'U238', 'U238', 'U',
    'Be', 'Be9', 'Be', 'Be9', 'Be', 'Be9'
]

str_mcnp_cases = [
    '1000', '1000', '1001', '20000', '20000', '20000.21c', '20040.21c', '20040', '20042.21c', '20043', '20041',
    '92000', '92000', '92235', '92235.50c', '92238', '92238', '92000', '4000', '4009', '4000', '4009', '4000', '4009'
]

# Equality matrix for creation_cases
equality = [
    [1, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [1, 1, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 1,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    
    [0, 0, 0,  1, 1, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  1, 1, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 1, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 1, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 1, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 1, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 1, 1, 0,  0, 0, 0, 0, 0, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  1, 1, 0, 0, 0, 0, 1,  0, 0, 0, 0, 0, 0],
    
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  1, 0, 1, 0, 1, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 1, 0, 1, 0, 1],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  1, 0, 1, 0, 1, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 1, 0, 1, 0, 1],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  1, 0, 1, 0, 1, 0],
    [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 1, 0, 1, 0, 1]
]
