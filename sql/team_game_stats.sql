CREATE TABLE TeamGameStats (
    Season      INTEGER,
    DayNum      INTEGER,
    TeamID      INTEGER,
    OppTeamID   INTEGER,
    GameType    TEXT,     -- 'RegularSeason' or 'Tournament'
    Score       INTEGER,  -- Team's points scored
    OppScore    INTEGER,  -- Opponent's points scored
    FGM         INTEGER,  -- Field Goals Made by the team
    FGA         INTEGER,  -- Field Goals Attempted by the team
    FGM3        INTEGER,  -- 3-point Field Goals Made by the team
    FGA3        INTEGER,  -- 3-point Field Goals Attempted by the team
    FTM         INTEGER,  -- Free Throws Made by the team
    FTA         INTEGER,  -- Free Throws Attempted by the team
    "OR"          INTEGER,  -- Offensive Rebounds by the team
    DR          INTEGER,  -- Defensive Rebounds by the team
    Ast         INTEGER,  -- Assists by the team
    "TO"          INTEGER,  -- Turnovers by the team
    Stl         INTEGER,  -- Steals by the team
    Blk         INTEGER,  -- Blocks by the team
    PF          INTEGER,  -- Personal fouls by the team
    OppFGM      INTEGER,  -- Field Goals Made by the opponent
    OppFGA      INTEGER,  -- Field Goals Attempted by the opponent
    OppFGM3     INTEGER,  -- 3-point FG Made by the opponent
    OppFGA3     INTEGER,  -- 3-point FG Attempted by the opponent
    OppFTM      INTEGER,  -- Free Throws Made by the opponent
    OppFTA      INTEGER,  -- Free Throws Attempted by the opponent
    OppOR       INTEGER,  -- Offensive Rebounds by the opponent
    OppDR       INTEGER,  -- Defensive Rebounds by the opponent
    OppAst      INTEGER,  -- Assists by the opponent
    OppTO       INTEGER,  -- Turnovers by the opponent
    OppStl      INTEGER,  -- Steals by the opponent
    OppBlk      INTEGER,  -- Blocks by the opponent
    OppPF       INTEGER,  -- Personal fouls by the opponent
    NumOT       INTEGER,  -- Number of overtime periods
    PRIMARY KEY (Season, DayNum, TeamID)
);