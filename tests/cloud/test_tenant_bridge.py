"""Phase A7 — tenant bridge."""
from __future__ import annotations

import uuid

from server.cloud.auth.context import AuthContext
from server.cloud.auth.tenant_bridge import to_tenant_context


def test_to_tenant_context_round_trip():
    auth = AuthContext(
        user_id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        tenant_id="org_abc",
        sub_tenant_id="sub_xyz",
        role="OWNER",
        api_key_id=None,
        auth_method="clerk_jwt",
    )
    ctx = to_tenant_context(auth)
    assert ctx.tenant_id == "org_abc"
    assert ctx.sub_tenant_id == "sub_xyz"
    assert ctx.user_id == str(auth.user_id)
    assert ctx.role == "OWNER"
    assert ctx.agent_id is None
    assert ctx.run_id is None
