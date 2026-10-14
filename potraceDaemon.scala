import java.nio.file._, StandardWatchEventKinds._
import javax.imageio.ImageIO
import scala.jdk.CollectionConverters._
import scala.sys.process._
import java.io.FileOutputStream
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

@main def potraceDaemon(inDir: String, outDir: String): Unit =
  val (in, out) = (Paths.get(inDir), Paths.get(outDir))
  val watcher = in.getFileSystem.newWatchService()
  in.register(watcher, ENTRY_CREATE); Files.createDirectories(out)
  
  println(s"[*] Edge detection daemon started. Watching: $in -> Output: $out")
  
  while true do
    val key = watcher.take()
    for e <- key.pollEvents().asScala if e.kind() == ENTRY_CREATE do
      val file = in.resolve(e.context.asInstanceOf[Path])
      if Files.isRegularFile(file) then
        println(s"[*] Detected new file: ${file.getFileName}. Waiting for write completion...")
        
        // Wait for file size to stabilize
        var lastSize = -1L
        var currentSize = Files.size(file)
        var checks = 0
        while currentSize != lastSize && checks < 20 do
          lastSize = currentSize
          Thread.sleep(250)
          try currentSize = Files.size(file) catch case _ => ()
          checks += 1

        println(s"[*] Processing image: ${file.getFileName}...")
        
        // Safely decode with a 5-second timeout to bypass Java ImageIO infinite loops on bad JPEGs
        val decodeFuture = Future {
          Option(ImageIO.read(file.toFile))
        }

        try
          Await.result(decodeFuture, 5.seconds) match {
            case Some(img) =>
              val (w, h) = (img.getWidth, img.getHeight)
              println(s"[*] Image loaded: ${w}x${h} pixels. Bulk extracting pixel buffer...")
              
              val pixels = img.getRGB(0, 0, w, h, null, 0, w)
              println(s"[*] Pixel buffer extracted. Computing gradients & writing binary PBM...")
              
              val pbm = Files.createTempFile("edge", ".pbm")
              val outStream = new FileOutputStream(pbm.toFile)
              try
                outStream.write(s"P4\n$w $h\n".getBytes("US-ASCII"))
                for y <- 0 until h do
                  var byteVal = 0
                  var bitCount = 0
                  val rowOffset = y * w
                  for x <- 0 until w do
                    val gx = if x > 0 && x < w - 1 then (pixels(rowOffset + x + 1) & 0xff) - (pixels(rowOffset + x - 1) & 0xff) else 0
                    val gy = if y > 0 && y < h - 1 then (pixels((y + 1) * w + x) & 0xff) - (pixels((y - 1) * w + x) & 0xff) else 0
                    val bit = if math.hypot(gx, gy) > 60 then 1 else 0
                    byteVal = (byteVal << 1) | bit
                    bitCount += 1
                    if bitCount == 8 then
                      outStream.write(byteVal)
                      byteVal = 0
                      bitCount = 0
                  if bitCount > 0 then
                    outStream.write(byteVal << (8 - bitCount))
              finally outStream.close()
              
              println(s"[*] PBM generated. Running potrace vectorization...")
              val svg = Files.createTempFile("edge", ".svg")
              s"potrace -s -o ${svg.toAbsolutePath} ${pbm.toAbsolutePath}".!
              
              val outFile = out.resolve(s"${file.getFileName}.svg")
              Files.copy(svg, outFile, StandardCopyOption.REPLACE_EXISTING)
              Files.deleteIfExists(pbm); Files.deleteIfExists(svg)
              println(s"[+] Successfully generated SVG: ${file.getFileName}.svg")
              
            case None =>
              println(s"[!] Warning: ImageIO returned null for ${file.getFileName}. Skipping.")
          }
        catch
          case _: TimeoutException =>
            println(s"[!] Error: Image decoding timed out for ${file.getFileName} (malformed/progressive JPEG). Skipping file.")
          case e: Exception =>
            println(s"[!] Error processing ${file.getFileName}: ${e.getMessage}")
            
    key.reset()