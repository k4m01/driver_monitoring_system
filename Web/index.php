    <?php
    $servername = "localhost";
    $username = "root";
    $password = "";
    $dbname = "alert_db";

    $conn = new mysqli($servername, $username, $password, $dbname);

    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    $sql = "SELECT * FROM alerts ORDER BY timestamp DESC";
    $result = $conn->query($sql);
    ?>

    <!DOCTYPE html>
    <html>
    <head>
        <title>Alert Images</title>
    </head>
    <body>
        <h1>Captured Alerts</h1>
        <table border="1">
            <tr>
                <th>Image</th>
                <th>Type</th>
                <th>Timestamp</th>
            </tr>
            <?php
            if ($result->num_rows > 0) {
                while ($row = $result->fetch_assoc()) {
                    echo "<tr>";
                    if (isset($row['filename']) && !empty($row['filename'])) {
                        // ใช้พาธแบบสัมพันธ์กับตำแหน่งใน htdocs
                        $image_path = "alert_images/" . $row['filename'];
                        echo "<td><img src='" . $image_path . "' width='150'></td>";
                    } else {
                        echo "<td>No Image Available</td>";
                    }
                    echo "<td>" . $row['alert_type'] . "</td>";
                    echo "<td>" . $row['timestamp'] . "</td>";
                    echo "</tr>";
                }
            } else {    
                echo "<tr><td colspan='3'>No alerts captured</td></tr>";
            }
            ?>
        </table>
    </body>
    </html>
