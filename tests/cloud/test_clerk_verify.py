"""Phase A5 — Clerk JWT verifier.

Generate an RSA key pair in-test, monkeypatch ``_get_jwks`` to return
the public half, then sign + verify tokens. Cover the happy path,
expired tokens, wrong issuer, and the unknown-kid → refresh path.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt
from jose.utils import long_to_base64


def _rsa_keys():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": "test-kid",
        "alg": "RS256",
        "use": "sig",
        "n": long_to_base64(public_numbers.n).decode("ascii"),
        "e": long_to_base64(public_numbers.e).decode("ascii"),
    }
    pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem, jwk


def _make_token(pem: bytes, *, iss: str, sub: str, exp_offset: int = 60):
    now = int(time.time())
    return jwt.encode(
        {
            "sub": sub,
            "email": "x@example.com",
            "iss": iss,
            "iat": now,
            "exp": now + exp_offset,
        },
        pem,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )


@pytest.mark.asyncio
async def test_verify_happy_path(cloud_env, monkeypatch):
    pem, jwk = _rsa_keys()
    from server.cloud.auth import clerk

    async def _fake_jwks(force_refresh=False):
        return {"keys": [jwk]}

    monkeypatch.setattr(clerk, "_get_jwks", _fake_jwks)
    token = _make_token(
        pem, iss="https://example.clerk.accounts.dev/", sub="user_abc"
    )
    claims = await clerk.verify_clerk_jwt(token)
    assert claims.sub == "user_abc"
    assert claims.email == "x@example.com"


@pytest.mark.asyncio
async def test_verify_expired_token(cloud_env, monkeypatch):
    pem, jwk = _rsa_keys()
    from server.cloud.auth import clerk

    async def _fake_jwks(force_refresh=False):
        return {"keys": [jwk]}

    monkeypatch.setattr(clerk, "_get_jwks", _fake_jwks)
    token = _make_token(
        pem, iss="https://example.clerk.accounts.dev/", sub="x", exp_offset=-1
    )
    with pytest.raises(clerk.ClerkAuthError):
        await clerk.verify_clerk_jwt(token)


@pytest.mark.asyncio
async def test_verify_wrong_issuer(cloud_env, monkeypatch):
    pem, jwk = _rsa_keys()
    from server.cloud.auth import clerk

    async def _fake_jwks(force_refresh=False):
        return {"keys": [jwk]}

    monkeypatch.setattr(clerk, "_get_jwks", _fake_jwks)
    token = _make_token(pem, iss="https://attacker.example/", sub="x")
    with pytest.raises(clerk.ClerkAuthError):
        await clerk.verify_clerk_jwt(token)


@pytest.mark.asyncio
async def test_verify_unknown_kid_triggers_refresh(cloud_env, monkeypatch):
    pem, jwk = _rsa_keys()
    from server.cloud.auth import clerk

    refresh_calls = {"count": 0}

    async def _fake_jwks(force_refresh=False):
        refresh_calls["count"] += 1
        if refresh_calls["count"] == 1:
            return {"keys": [{**jwk, "kid": "stale-kid"}]}
        return {"keys": [jwk]}

    monkeypatch.setattr(clerk, "_get_jwks", _fake_jwks)
    token = _make_token(
        pem, iss="https://example.clerk.accounts.dev/", sub="user_xyz"
    )
    claims = await clerk.verify_clerk_jwt(token)
    assert claims.sub == "user_xyz"
    assert refresh_calls["count"] == 2  # stale + force refresh


def test_clear_jwks_cache(cloud_env):
    from server.cloud.auth import clerk

    clerk._jwks_cache["fetched_at"] = time.time()
    clerk._jwks_cache["keys"] = {"keys": []}
    clerk.clear_jwks_cache()
    assert clerk._jwks_cache["keys"] is None
    assert clerk._jwks_cache["fetched_at"] == 0.0
