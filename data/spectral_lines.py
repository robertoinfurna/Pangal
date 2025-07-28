spectral_lines = {
    "Hd": 4102,
    "Hg": 4340,
    "Hb": 4861,
    "Ha": 6563,
    "He II 4686": 4686,
    "He I 4471": 4471,
    "He I 5876": 5876,
    "[O II] 3727": 3727,
    "[O III] 4959": 4959,
    "[O III] 5007": 5007,
    "[N II] 6548": 6548,
    "[N II] 6583": 6583,
    "[S II] 6716": 6716,
    "[S II] 6731": 6731,
    "[Ar III] 7136": 7136,
    "[Fe II] 5159": 5159,
    "[Fe I] 4383": 4383,
    "[Fe I] 5270": 5270,
    "[Fe I] 5335": 5335,
    "[Fe VII] 6087": 6087,
    "[Ca II] 8498": 8498,
    "[Ca II] 8542": 8542,
    "[Ca II] 8662": 8662,
    "Ca H": 3968,
    "Ca K": 3934,
    "Mg I 5167": 5167,
    "Mg I 5172": 5172,
    "Mg I 5183": 5183,
    "Na I D1": 5890,  # First sodium doublet component
    "Na I D2": 5896,  # Second sodium doublet component
    "G-band (CH) 4300": 4300,
    "CN Band 4215": 4215,
}

# Define UBGVRI filter wavelength ranges (in Angstrom)
UBGVRI_filters = {
    "U": {"pivot_wavelength": 3650, "bandwidth": 660, "color": "purple"},
    "B": {"pivot_wavelength": 4450, "bandwidth": 940, "color": "blue"},
    "G": {"pivot_wavelength": 4640, "bandwidth": 1280, "color": "green"},
    "V": {"pivot_wavelength": 5510, "bandwidth": 880, "color": "yellowgreen"},
    "R": {"pivot_wavelength": 6580, "bandwidth": 1380, "color": "red"},
    "I": {"pivot_wavelength": 8060, "bandwidth": 1490, "color": "darkred"},
}


atmospheric_lines = {
    # Oxygen Absorption Bands
    "O₂ A-band": 7605,
    "O₂ B-band": 6870,
    "O₂ γ-band": 6280,

    # Water Vapor (H₂O) Absorption Bands
    #"H₂O 5900-7300": "5900-7300",
    #"H₂O 8200-9800": "8200-9800",

    # Ozone (O₃) Absorption
    #"O₃ Chappuis Band": "4500-7500",
    #"O₃ Huggins Band": "<3400",

    # Atmospheric Emission (Airglow)
    "[O I] 5577 (Green Airglow)": 5577,
    "[O I] 6300": 6300,
    "[O I] 6364": 6364,
    
    # Hydroxyl (OH) Bands (dominant night-sky emission)
    #"OH 6500-9000": "6500-9000",

    # Sodium Emission (Mesospheric)
    "Na I D1": 5890,
    "Na I D2": 5896,

    # Nitrogen Emission
    #"N₂ First Positive Band": "5500-7000",
}

