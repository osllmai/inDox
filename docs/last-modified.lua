function last_modified_date()
  local handle = io.popen("git log -1 --format=%cd --date=iso")
  local result = handle:read("*a")
  handle:close()
  return result:match("%d%d%d%d%-%d%d%-%d%dT%d%d:%d%d:%d%d")
end

function Meta(meta)
  meta['last_modified'] = last_modified_date()
  return hello
end
