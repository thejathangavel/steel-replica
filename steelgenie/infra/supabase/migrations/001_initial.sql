-- =============================================================================
-- SteelGenie — Initial Schema Migration
-- Run this in: Supabase Dashboard → SQL Editor → New Query → Run
-- =============================================================================

-- ---------------------------------------------------------------------------
-- TABLES
-- ---------------------------------------------------------------------------

CREATE TABLE projects (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          UUID REFERENCES auth.users NOT NULL,
  name             TEXT NOT NULL,
  project_number   TEXT,
  design_standard  TEXT DEFAULT 'AISC',
  unit_system      TEXT DEFAULT 'imperial',
  location         TEXT,
  status           TEXT DEFAULT 'in_progress',
  created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE drawings (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id     UUID REFERENCES projects NOT NULL,
  filename       TEXT NOT NULL,
  r2_key         TEXT NOT NULL,
  page_count     INT,
  pdf_type       TEXT,
  drawing_scale  TEXT,
  page_status    TEXT DEFAULT 'uploaded',
  status         TEXT DEFAULT 'uploaded',
  created_at     TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE members (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  drawing_id   UUID REFERENCES drawings NOT NULL,
  page_number  INT NOT NULL,
  profile      TEXT NOT NULL,
  standard     TEXT DEFAULT 'AISC',
  member_type  TEXT,
  length_ft    FLOAT,
  confidence   FLOAT,
  bbox         JSONB,
  verified     BOOLEAN DEFAULT false,
  created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE jobs (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  drawing_id  UUID REFERENCES drawings NOT NULL,
  job_type    TEXT NOT NULL,
  status      TEXT DEFAULT 'queued',
  progress    INT DEFAULT 0,
  error       TEXT,
  result      JSONB,
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- ROW LEVEL SECURITY — Enable on all tables
-- ---------------------------------------------------------------------------

ALTER TABLE projects  ENABLE ROW LEVEL SECURITY;
ALTER TABLE drawings  ENABLE ROW LEVEL SECURITY;
ALTER TABLE members   ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs      ENABLE ROW LEVEL SECURITY;

-- ---------------------------------------------------------------------------
-- RLS POLICIES
--
-- Ownership chain:
--   jobs → drawings → projects → auth.users (user_id)
--
-- Each policy uses a subquery join to verify the authenticated user
-- owns the parent project, so data never leaks across users.
-- ---------------------------------------------------------------------------

-- projects: direct ownership check
CREATE POLICY own_projects ON projects
  FOR ALL
  USING (auth.uid() = user_id);

-- drawings: must belong to one of the user's projects
CREATE POLICY own_drawings ON drawings
  FOR ALL
  USING (
    EXISTS (
      SELECT 1
      FROM   projects
      WHERE  projects.id      = drawings.project_id
        AND  projects.user_id = auth.uid()
    )
  );

-- members: must belong to a drawing in one of the user's projects
CREATE POLICY own_members ON members
  FOR ALL
  USING (
    EXISTS (
      SELECT 1
      FROM   drawings
      JOIN   projects ON projects.id = drawings.project_id
      WHERE  drawings.id      = members.drawing_id
        AND  projects.user_id = auth.uid()
    )
  );

-- jobs: same chain as members
CREATE POLICY own_jobs ON jobs
  FOR ALL
  USING (
    EXISTS (
      SELECT 1
      FROM   drawings
      JOIN   projects ON projects.id = drawings.project_id
      WHERE  drawings.id      = jobs.drawing_id
        AND  projects.user_id = auth.uid()
    )
  );
