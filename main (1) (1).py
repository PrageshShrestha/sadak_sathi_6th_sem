from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, validator
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from database import SessionLocal, engine, Base, get_db
import models
import traceback
import xgboost as xgb
import requests, os
import secrets
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
from math import radians, cos, sin, asin, sqrt
from typing import List

app = FastAPI(title="Sadak-Sathi API")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/model_xgb.pkl"
app.state.xgb_model: xgb.Booster | None = None

@app.on_event("startup")
async def on_startup():
    print("Starting up Sadak-Sathi Backend...")
    Base.metadata.create_all(bind=engine)
    print("SQLite database 'sadak_sathi.db' ready and tables created!")

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at '{MODEL_PATH}'")
        print(" → ETA prediction will be disabled until model is placed correctly.")
        app.state.xgb_model = None
    else:
        try:
            model = xgb.Booster()
            model.load_model(MODEL_PATH)
            app.state.xgb_model = model
            print(f"XGBoost model loaded successfully from '{MODEL_PATH}'")
            print(" → Ready for fast ETA predictions!")
        except Exception as e:
            print(f"ERROR loading XGBoost model: {e}")
            app.state.xgb_model = None

    print("Backend fully started and ready!\n")

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
SECRET_KEY = "your-super-secret-key-2025-change-in-production-change-this"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY")
MAILGUN_DOMAIN = "sandboxc91dffeec1b34d0d900ddcc1d515312d.mailgun.org"
MAILGUN_URL = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages"

def get_password_hash(pw: str):
    return pwd_context.hash(pw)

def verify_password(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401)
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=401)
    return user

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

    @validator("password")
    def password_min_length(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    is_admin: bool = False

    class Config:
        from_attributes = True

class AdminRequest(BaseModel):
    username: str
    email: EmailStr

class AdminComplete(BaseModel):
    token: str
    password: str
    name: str

def send_verification_email(email: str, username: str, token: str):
    link = f"http://localhost:8000/verify-admin/{token}"
    try:
        requests.post(
            MAILGUN_URL,
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": f"Sadak-Sathi Admin <admin@{MAILGUN_DOMAIN}>",
                "to": email,
                "subject": "Verify Admin Account - Sadak-Sathi",
                "html": f"""
                <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;padding:30px;background:#f9f9f9;border-radius:12px;text-align:center">
                <h2 style="color:#1E88E5">Hello {username}!</h2>
                <p>You requested admin access.</p>
                <p><strong>Click below to verify your email and set password:</strong></p>
                <a href="{link}" style="background:#1E88E5;color:white;padding:16px 40px;text-decoration:none;border-radius:12px;font-size:18px;display:inline-block;margin:20px 0">
                    Verify Email & Set Password
                </a>
                <p>Copy this Link: <code>{link}</code></p>
                </div>
                """
            }
        )
        print(f"Email sent to {email}")
    except Exception as e:
        print(f"Mailgun failed: {e}")

@app.post("/admin/request-verification")
def request_admin_verification(payload: AdminRequest, background: BackgroundTasks, db: Session = Depends(get_db)):
    email = payload.email
    username = payload.username
    if db.query(models.User).filter(models.User.email == email, models.User.is_admin == True).first():
        raise HTTPException(status_code=400, detail="Already an admin")
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=1)
    old = db.query(models.AdminVerificationToken).filter(models.AdminVerificationToken.email == email).first()
    if old:
        db.delete(old)
        db.commit()
    verification = models.AdminVerificationToken(email=email, token=token, expires_at=expires_at)
    db.add(verification)
    db.commit()
    background.add_task(send_verification_email, email, username, token)
    return {"message": "Verification email sent! Check your inbox."}

@app.get("/verify-admin/{token}")
def verify_admin(token: str, db: Session = Depends(get_db)):
    verification = db.query(models.AdminVerificationToken).filter(
        models.AdminVerificationToken.token == token,
        models.AdminVerificationToken.used == False,
        models.AdminVerificationToken.expires_at > datetime.utcnow()
    ).first()
    if not verification:
        return HTMLResponse(content="""
        <html>
          <body style="font-family:Arial;text-align:center;padding:100px;background:#ffebee">
            <h2 style="color:#c62828">Invalid or Expired Link</h2>
            <p>Please request a new verification email.</p>
          </body>
        </html>
        """)
    verification.used = True
    db.commit()
    return HTMLResponse(content="""
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sadak-Sathi Admin Verified</title>
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&display=swap');
          body { margin:0; padding:0; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family:'Poppins',Arial,sans-serif; }
          .container { min-height:100vh; display:flex; align-items:center; justify-content:center; }
          .card { background:white; padding:50px 40px; border-radius:20px; box-shadow:0 20px 60px rgba(0,0,0,0.3); text-align:center; max-width:90%; width:500px; }
          .check { width:120px; height:120px; background:#1E88E5; border-radius:50%; margin:0 auto 30px; display:flex; align-items:center; justify-content:center; }
          svg { width:80px; height:80px; }
          h1 { color:#1E88E5; font-size:32px; margin:0 0 20px; }
          p { color:#555; font-size:18px; line-height:1.6; }
          .box { background:#f0f8ff; padding:20px; border-radius:12px; margin:30px 0; }
          .box p { color:#1E88E5; font-weight:600; margin:0; }
          small { color:#888; font-size:14px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="card">
            <div class="check">
              <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3">
                <path d="M20 6L9 17l-5-5"/>
              </svg>
            </div>
            <h1>Email Verified Successfully!</h1>
            <p>Welcome to <strong>Sadak-Sathi Admin Panel</strong><br>
            You can now set your password and start managing buses.</p>
            <div class="box">
              <p>Close this tab and return to the app</p>
            </div>
            <small>© 2025 Sadak-Sathi • Nepal's Smart Bus Tracker</small>
            <script>
              setTimeout(() => window.close(), 4000);
            </script>
          </div>
        </div>
      </body>
    </html>
    """)

@app.post("/admin/complete-registration")
def complete_admin_registration(payload: AdminComplete, db: Session = Depends(get_db)):
    token = payload.token
    password = payload.password
    name = payload.name.strip() or "Admin"
    verification = db.query(models.AdminVerificationToken).filter(
        models.AdminVerificationToken.token == token,
        models.AdminVerificationToken.used == True
    ).first()
    if not verification:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    user = db.query(models.User).filter(models.User.email == verification.email).first()
    if user:
        user.hashed_password = get_password_hash(password)
        user.is_admin = True
        user.name = name
    else:
        user = models.User(
            name=name,
            email=verification.email,
            hashed_password=get_password_hash(password),
            is_admin=True
        )
        db.add(user)
    db.commit()
    return {"message": "Admin created! Login now"}

@app.get("/check-admin-verification/{email}")
def check_admin_verification(email: str, db: Session = Depends(get_db)):
    verification = db.query(models.AdminVerificationToken).filter(
        models.AdminVerificationToken.email == email,
        models.AdminVerificationToken.used == True
    ).first()
    return {"verified": verification is not None, "token": verification.token if verification else None}

@app.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = get_password_hash(user_in.password)
    new_user = models.User(name=user_in.name, email=user_in.email, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": user.email})
    return {"access_token": token}

@app.get("/me", response_model=UserResponse)
def me(current_user: models.User = Depends(get_current_user)):
    return current_user

@app.post("/change-password")
def change_password(payload: dict, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    old = payload.get("old_password")
    new = payload.get("new_password")
    if not old or not new:
        raise HTTPException(status_code=400, detail="Missing passwords")
    if not verify_password(old, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    current_user.hashed_password = get_password_hash(new)
    db.commit()
    return {"message": "Password changed successfully"}

@app.delete("/delete-account")
def delete_account(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(current_user)
    db.commit()
    return {"message": "Account deleted permanently"}

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return c * r

@app.post("/find-stops")
async def find_stops(request: dict):
    route_coords = request.get("route", [])
    if len(route_coords) < 2:
        return []
    simplified = route_coords[::10] if len(route_coords) > 50 else route_coords
    lats = [c[1] for c in simplified]
    lons = [c[0] for c in simplified]
    min_lat, max_lat = min(lats) - 0.01, max(lats) + 0.01
    min_lon, max_lon = min(lons) - 0.01, max(lons) + 0.01
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["highway"="bus_stop"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["public_transport"="platform"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    """
    try:
        resp = requests.post("https://overpass-api.de/api/interpreter", data={'data': overpass_query}, timeout=25)
        if resp.status_code != 200:
            return []
        elements = resp.json().get("elements", [])
        stops = []
        seen = set()
        for element in elements:
            if element["id"] in seen:
                continue
            lat = element["lat"]
            lon = element["lon"]
            name = element.get("tags", {}).get("name", "Bus Stop")
            close = any(haversine(lon, lat, c[0], c[1]) <= 50 for c in route_coords)
            if close:
                stops.append({"name": name, "lat": lat, "lon": lon})
                seen.add(element["id"])
        return stops
    except Exception as e:
        print(f"Overpass fallback: {e}")
        return []

@app.post("/admin/save-route")
async def save_new_route(route_data: dict, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Only admins can save routes")
    try:
        route_name = route_data.get("route_name")
        coordinates = route_data.get("coordinates")
        stops = route_data.get("stops")
        if not route_name or not isinstance(coordinates, list) or not isinstance(stops, list):
            raise HTTPException(status_code=400, detail="Invalid payload: 'route_name'(str), 'coordinates'(list), 'stops'(list) required")
        for c in coordinates:
            if not isinstance(c, (list, tuple)) or len(c) != 2:
                raise HTTPException(status_code=400, detail="Invalid coordinates format. Expected list of [lon, lat].")
        coordinates_json = json.dumps(coordinates)
        stops_json = json.dumps(stops)
        new_route = models.RouteInfo(
            route_name=route_name,
            coordinates=coordinates_json,
            stops=stops_json
        )
        db.add(new_route)
        db.commit()
        db.refresh(new_route)
        return {"id": new_route.id, "message": f"Route '{route_name}' saved successfully with {len(stops)} stops!"}
    except HTTPException:
        raise
    except IntegrityError:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Route with the same name already exists")
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Save error: {str(e)}")

@app.get("/routes")
def get_all_routes(db: Session = Depends(get_db)):
    routes = db.query(models.RouteInfo).all()
    return [
        {
            "id": r.id,
            "route_name": r.route_name,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in routes
    ]

@app.get("/route/{route_id}")
def get_route(route_id: int, db: Session = Depends(get_db)):
    route = db.query(models.RouteInfo).filter(models.RouteInfo.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    return {
        "id": route.id,
        "route_name": route.route_name,
        "coordinates": json.loads(route.coordinates),
        "stops": json.loads(route.stops),
        "created_at": route.created_at.isoformat() if route.created_at else None,
    }

@app.put("/admin/update-route/{route_id}")
async def update_route(route_id: int, route_data: dict, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    route = db.query(models.RouteInfo).filter(models.RouteInfo.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    route.coordinates = json.dumps(route_data["coordinates"])
    route.stops = json.dumps(route_data["stops"])
    db.commit()
    return {"message": f"Route '{route.route_name}' updated successfully!"}

@app.delete("/admin/delete-route/{route_id}")
async def delete_route(route_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    route = db.query(models.RouteInfo).filter(models.RouteInfo.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    db.delete(route)
    db.commit()
    return {"message": f"Route '{route.route_name}' deleted successfully!"}

class EtaInput(BaseModel):
    from_stop: int
    to_stop: int
    dist: float
    dir_ref: int
    rush: int

@app.post("/predict-eta")
async def predict_eta(input: EtaInput):
    model: xgb.Booster = app.state.xgb_model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ETA prediction is currently unavailable. Model not loaded (check server startup logs)."
        )
    try:
        features = [[
            input.from_stop,
            input.to_stop,
            input.dist,
            input.dir_ref,
            input.rush
        ]]
        FEATURE_NAMES = ["from", "to", "dist", "dir_ref", "rush"]
        dmatrix = xgb.DMatrix(features, feature_names=FEATURE_NAMES)
        prediction = model.predict(dmatrix)[0]
        eta_seconds = float(prediction)
        return {"eta_seconds": eta_seconds}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Failed to generate ETA prediction. Please try again later."
        )

# ============================
# REALISTIC RING ROAD SIMULATION – LINEAR SMOOTH MOVEMENT
# ============================

BASE_STOPS_ORDER = [
    "koteshwor","airport","gausala","chabhil","gangabu",
    "samakhushi","balaju","banasthali","swoyambhu",
    "satdobato","gwarko","balkumari"
]
NUM_STOPS = len(BASE_STOPS_ORDER)

WAITING_TIME_TABLE = {  # stop_index -> {rush: seconds}
    0: {0: 102, 1: 107, 2: 101},
    1: {0: 69, 1: 87, 2: 62},
    2: {0: 212, 1: 225, 2: 203},
    3: {0: 135, 1: 162, 2: 132},
    4: {0: 60, 1: 109, 2: 56},
    5: {0: 91, 1: 110, 2: 83},
    6: {0: 115, 1: 114, 2: 125},
    7: {0: 63, 1: 69, 2: 70},
    8: {0: 146, 1: 149, 2: 150},
    9: {0: 267, 1: 258, 2: 262},
    10: {0: 136, 1: 142, 2: 138},
    11: {0: 73, 1: 73, 2: 83},
}

def get_rush_type(now: datetime) -> int:
    hour = now.hour
    if 8 <= hour <= 11: return 1  # morning rush
    if 16 <= hour <= 19: return 2  # evening rush
    return 0  # normal

class BusState:
    def __init__(self, bus_number: str, direction: int):
        self.bus_number = bus_number
        self.direction = direction  # 0 = clockwise, 1 = anticlockwise
        self.current_stop_idx = int(bus_number.split("-")[-1]) % NUM_STOPS
        self.time_to_next = 300.0
        self.initial_time_to_next = 300.0  # stores the full ETA for the current segment
        self.waiting_remaining = 0.0
        self.path_coords = []

simulation_buses: List[BusState] = []

@app.on_event("startup")
async def init_simulation():
    global simulation_buses
    simulation_buses.clear()
    for i in range(8):
        direction = 0 if i < 4 else 1
        bus = BusState(f"Sim-Bus-{i+1:02d}", direction)
        simulation_buses.append(bus)
    print("Simulation initialized: 4 clockwise + 4 anticlockwise buses")

class SimulatedBusResponse(BaseModel):
    bus_number: str
    lat: float
    lon: float
    next_stop_name: str
    eta_seconds_to_next: float
    waiting_time_at_current: int
    direction: str

class SimulatedStopResponse(BaseModel):
    name: str
    lat: float
    lon: float
    upcoming_buses: List[dict]

class RingRoadSimulationResponse(BaseModel):
    route: dict
    buses: List[SimulatedBusResponse]
    stops: List[SimulatedStopResponse]
@app.get("/map/simulate/ring-road", response_model=RingRoadSimulationResponse)
def simulate_ring_road(db: Session = Depends(get_db)):
    global simulation_buses
    try:
        route = db.query(models.RouteInfo).filter(models.RouteInfo.route_name == "Ring-Road-Ktm").first()
        if not route:
            raise HTTPException(404, "Ring Road route not found")

        full_coordinates = json.loads(route.coordinates)
        stops_raw = json.loads(route.stops)

        # Ensure the route is a closed loop (common for ring roads)
        # If first and last coordinates are not the same, append the first to close it
        if full_coordinates and full_coordinates[0] != full_coordinates[-1]:
            full_coordinates = full_coordinates + [full_coordinates[0]]

        # Map stop names → exact coordinates + closest path index (full scan)
        stop_map = {}
        stop_path_indices = []
        for i, name in enumerate(BASE_STOPS_ORDER):
            target_lat, target_lon = None, None
            best_idx = 0
            min_dist = float("inf")

            # Find matching stop in stops_raw by name
            matched_stop = next(
                (s for s in stops_raw if (s[0] if isinstance(s, list) else s.get("name", "")).lower() == name.lower()),
                None
            )

            if matched_stop:
                s_lat = matched_stop[1] if isinstance(matched_stop, list) else matched_stop.get("lat")
                s_lon = matched_stop[2] if isinstance(matched_stop, list) else matched_stop.get("lon")
                target_lat, target_lon = s_lat, s_lon

                # Snap to closest point on the (now closed) polyline
                for j, coord in enumerate(full_coordinates):
                    d = haversine(coord[0], coord[1], s_lon, s_lat)
                    if d < min_dist:
                        min_dist = d
                        best_idx = j
            else:
                # No name match – fallback to reasonable position (e.g., even spacing or first point)
                print(f"Warning: No stop data found for '{name}'. Using path index fallback.")
                best_idx = int(i / NUM_STOPS * len(full_coordinates))  # Rough proportional position
                target_lat = full_coordinates[best_idx][1]
                target_lon = full_coordinates[best_idx][0]

            stop_map[name] = {"lat": target_lat, "lon": target_lon}
            stop_path_indices.append(best_idx)

            # Debug output – remove in production if desired
            print(f"Stop '{name}' (idx {i}): path_index={best_idx}, coord=({target_lon:.6f}, {target_lat:.6f})")

        model = app.state.xgb_model
        now = datetime.now()
        rush = get_rush_type(now)

        bus_responses = []
        stop_upcoming = {i: [] for i in range(NUM_STOPS)}

        for bus in simulation_buses:
            # Use closed path for both directions
            bus.path_coords = full_coordinates if bus.direction == 0 else full_coordinates[::-1]

            current_idx = bus.current_stop_idx
            current_name = BASE_STOPS_ORDER[current_idx]

            # Normal progression
            delta = 1 if bus.direction == 0 else -1
            next_idx = (current_idx + delta) % NUM_STOPS

            # Clean loop restart for ring road
            if bus.direction == 0 and current_idx == NUM_STOPS - 1:   # Clockwise finished
                next_idx = 0
            elif bus.direction == 1 and current_idx == 0:            # Anticlockwise finished
                next_idx = NUM_STOPS - 1

            next_name = BASE_STOPS_ORDER[next_idx]
            current_stop = stop_map[current_name]
            next_stop = stop_map[next_name]

            if bus.waiting_remaining > 0:
                bus.waiting_remaining -= 1
                eta = bus.time_to_next
                wait_current = int(bus.waiting_remaining)
                lon, lat = current_stop["lon"], current_stop["lat"]
            else:
                if bus.time_to_next <= 0:
                    # Arrived → wait at next stop
                    wait_sec = WAITING_TIME_TABLE.get(current_idx, {0: 90, 1: 120, 2: 150}).get(rush, 90)
                    bus.waiting_remaining = wait_sec
                    bus.current_stop_idx = next_idx

                    dist_m = haversine(current_stop["lon"], current_stop["lat"],
                                      next_stop["lon"], next_stop["lat"])
                    if model and dist_m > 0:
                        try:
                            feat = [[current_idx, next_idx, dist_m, bus.direction, rush]]
                            dmat = xgb.DMatrix(feat, feature_names=["from", "to", "dist", "dir_ref", "rush"])
                            pred = model.predict(dmat)[0]
                            bus.time_to_next = max(60, float(pred))
                        except:
                            bus.time_to_next = max(180, dist_m / 20 / 1000 * 3600)
                    else:
                        bus.time_to_next = 300

                    bus.initial_time_to_next = bus.time_to_next
                    eta = bus.time_to_next
                    wait_current = int(wait_sec)
                    lon, lat = next_stop["lon"], next_stop["lat"]
                else:
                    bus.time_to_next -= 1
                    eta = bus.time_to_next
                    wait_current = 0

                    # Smooth linear movement along the path
                    progress = 0.0
                    if bus.initial_time_to_next > 0:
                        progress = 1 - (bus.time_to_next / bus.initial_time_to_next)

                    path_len = len(bus.path_coords)
                    start_path_idx = stop_path_indices[current_idx] if bus.direction == 0 else (path_len - 1 - stop_path_indices[current_idx])
                    end_path_idx = stop_path_indices[next_idx] if bus.direction == 0 else (path_len - 1 - stop_path_indices[next_idx])

                    target_idx = int(start_path_idx + progress * (end_path_idx - start_path_idx))
                    target_idx = max(0, min(path_len - 1, target_idx))
                    lon, lat = bus.path_coords[target_idx]

            bus_resp = SimulatedBusResponse(
                bus_number=bus.bus_number,
                lat=lat,
                lon=lon,
                next_stop_name=next_name,
                eta_seconds_to_next=round(eta),
                waiting_time_at_current=wait_current,
                direction="Clockwise" if bus.direction == 0 else "Anticlockwise"
            )
            bus_responses.append(bus_resp)

            stop_upcoming[next_idx].append({
                "bus_number": bus.bus_number,
                "eta_minutes": round(eta / 60, 1)
            })

        stop_info = []
        for i, name in enumerate(BASE_STOPS_ORDER):
            upcoming = sorted(stop_upcoming[i], key=lambda x: x["eta_minutes"])[:3]
            stop_info.append(SimulatedStopResponse(
                name=name,
                lat=stop_map[name]["lat"],
                lon=stop_map[name]["lon"],
                upcoming_buses=upcoming
            ))

        return RingRoadSimulationResponse(
            route={
                "id": route.id,
                "name": route.route_name,
                "coordinates": full_coordinates[:-1] if full_coordinates and full_coordinates[0] == full_coordinates[-1] else full_coordinates,  # Return open polyline to frontend if closed internally
                "stops": stops_raw
            },
            buses=bus_responses,
            stops=stop_info
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Simulation error: {traceback.format_exc()}")
        raise HTTPException(500, "Simulation failed")

@app.get("/map/route/{route_id}")
def give_route(route_id: int, db: Session = Depends(get_db)):
    try:
        route = db.query(models.RouteInfo).get(route_id)
        if not route:
            raise HTTPException(404, "Route not found")
        return {
    "id": route.id,
    "route_name": route.route_name,
    "coordinates": json.loads(route.coordinates),
    "stops": json.loads(route.stops),
    "created_at": route.created_at.isoformat() if route.created_at else None,
}
    except:
        raise HTTPException(500, "Failed to fetch route")


@app.get("/")
def root():
    return {"message": "Sadak-Sathi Backend - Running!"}