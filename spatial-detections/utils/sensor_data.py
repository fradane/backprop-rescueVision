import math


# PLACEHOLDER — in produzione questi valori arrivano da GPS e IMU reali
DRONE_LAT = 44.4056  # latitudine drone (placeholder: Genova)
DRONE_LON = 8.9463   # longitudine drone (placeholder: Genova)
DRONE_ALT = 50.0     # altitudine drone in metri (placeholder)
DRONE_HEADING = 180.0  # direzione camera in gradi (180 = Sud)
CAMERA_TILT = 45.0   # inclinazione camera verso il basso in gradi


def get_sensor_data() -> dict:
    """Restituisce i dati dei sensori. Sostituire con lettura reale da GPS/IMU."""
    return {
        "lat": DRONE_LAT,
        "lon": DRONE_LON,
        "alt": DRONE_ALT,
        "heading": DRONE_HEADING,
        "tilt": CAMERA_TILT,
    }


def compute_absolute_coordinates(
    x_mm: float, y_mm: float, z_mm: float, sensor: dict
) -> tuple[float, float]:
    """
    Converte le coordinate relative dalla camera (mm) in coordinate GPS assolute.

    x_mm: offset laterale (positivo = destra)
    y_mm: offset verticale (positivo = basso)
    z_mm: distanza frontale dalla camera

    La camera è inclinata di `tilt` gradi verso il basso rispetto all'orizzonte.
    """
    tilt_rad = math.radians(sensor["tilt"])
    heading_rad = math.radians(sensor["heading"])

    # converti mm in metri
    x_m = x_mm / 1000.0
    z_m = z_mm / 1000.0

    # proiezione sul piano orizzontale tenendo conto dell'inclinazione
    ground_dist = z_m * math.cos(tilt_rad)

    # offset nord/sud e est/ovest rispetto al drone
    delta_north = ground_dist * math.cos(heading_rad) + x_m * math.cos(heading_rad + math.pi / 2)
    delta_east  = ground_dist * math.sin(heading_rad) + x_m * math.sin(heading_rad + math.pi / 2)

    # converti metri in gradi (approssimazione piatta valida per piccole distanze)
    delta_lat = delta_north / 111320.0
    delta_lon = delta_east / (111320.0 * math.cos(math.radians(sensor["lat"])))

    abs_lat = sensor["lat"] + delta_lat
    abs_lon = sensor["lon"] + delta_lon

    return abs_lat, abs_lon
