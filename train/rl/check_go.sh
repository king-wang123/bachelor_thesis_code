#!/usr/bin/env bash
set -euo pipefail
ENV=/data/250010072/zlh/conda_envs/verl
export PATH="$ENV/bin:$PATH"
mkdir -p /tmp/go_check
cat > /tmp/go_check/main.go <<'GO'
package main
import "fmt"
func main(){var a,b int; fmt.Scan(&a,&b); fmt.Println(a+b)}
GO
go env GOCACHE GOPATH GOOS GOARCH
go build -x -o /tmp/go_check/main /tmp/go_check/main.go
echo "2 5" | /tmp/go_check/main

