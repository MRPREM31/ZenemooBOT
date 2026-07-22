"""
==============================================================================
Zenemoo AI - Authentication & Authorization Endpoints
==============================================================================
POST /auth/register
POST /auth/token
GET /auth/me
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from core.database import get_db
from core.security import verify_password, get_password_hash, create_access_token, decode_access_token
from api.models.user_model import User
from api.schemas.auth_schema import UserRegisterRequest, UserResponse, TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication & Security"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependency protecting private endpoints via JWT validation."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if not payload:
        raise credentials_exception
    
    username: str = payload.get("sub")
    if not username:
        raise credentials_exception

    stmt = select(User).where(User.username == username)
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    return user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserRegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Registers a new user account."""
    # Check existing username
    stmt = select(User).where(User.username == payload.username)
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{payload.username}' is already registered."
        )

    # Hash password & create user
    hashed_pwd = get_password_hash(payload.password)
    new_user = User(
        username=payload.username,
        email=payload.email,
        telegram_id=payload.telegram_id,
        hashed_password=hashed_pwd,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return new_user


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Authenticates user and returns JWT access token."""
    stmt = select(User).where(User.username == form_data.username)
    user = (await db.execute(stmt)).scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username, "user_id": user.id})
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in_minutes=1440,
        user=UserResponse.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Returns currently authenticated user profile."""
    return current_user
